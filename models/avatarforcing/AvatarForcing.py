import os, time
import torch, math
import numpy as np
import torch.nn as nn

import torch.nn.functional as F
import torchvision.transforms.functional as TF

from tqdm import tqdm
from PIL import ImageDraw, ImageFont

from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from models import BaseModel, BaseDiffusionModel, FlowMatchScheduler
from models.wav2vec2 import Wav2VecModel

from models.avatarforcing.generator import Generator as MotionAutoencoder
from models.avatarforcing.flow_transformer import FlowTransformer


################ Optional timing instrumentation (env-gated, off by default) ################
# Enable with AVATAR_TIMING=1. Adds torch.cuda.synchronize() around the hot spots, so it
# perturbs the pipeline slightly — never leave it on in production.
TIMING_ENABLED = os.environ.get("AVATAR_TIMING", "0") not in ("", "0", "false", "False", "no")
TIMING_STATS = {}


def _tic():
    if not TIMING_ENABLED:
        return None
    torch.cuda.synchronize()
    return time.perf_counter()


def _toc(key: str, t0):
    if t0 is None:
        return
    torch.cuda.synchronize()
    TIMING_STATS.setdefault(key, []).append((time.perf_counter() - t0) * 1000.0)


def timing_report(warmup: int = 1) -> str:
    """Human-readable ms table. `warmup` per-block samples are dropped from the averages."""
    if not TIMING_STATS:
        return "(AVATAR_TIMING not enabled / no samples)"
    lines = [f"{'stage':<34}{'n':>5}{'mean_ms':>10}{'min_ms':>9}{'max_ms':>9}"]
    for key in sorted(TIMING_STATS):
        vals = TIMING_STATS[key]
        used = vals[warmup:] if len(vals) > warmup + 1 else vals
        lines.append(f"{key:<34}{len(used):>5}{sum(used)/len(used):>10.1f}{min(used):>9.1f}{max(used):>9.1f}")
    return "\n".join(lines)


################ Encoders ################
class AudioEncoder(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # initialize audio encoder
        self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        self.wav2vec2.feature_extractor._freeze_parameters()
        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False

        # set num of frames
        self.only_last_features = opt.only_last_features        
        self.num_frames_for_clip = int(opt.sec * self.opt.fps)
        self.samples_per_frame = int(opt.sampling_rate / opt.fps)

        # set audio projection layers
        audio_input_dim = 768 if opt.only_last_features else 12 * 768
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_input_dim, opt.dim_w),
            nn.LayerNorm(opt.dim_w),
            nn.SiLU())

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features)
        if self.only_last_features:
            a = a.last_hidden_state
        else:
            a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
            a = a.reshape(a.shape[0], a.shape[1], -1)
        return a

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.shape[1] % int(self.num_frames_for_clip * self.samples_per_frame) != 0:
            a = F.pad(a, (0, int(self.num_frames_for_clip * self.samples_per_frame) - a.shape[1]), mode='replicate')
        a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)
        return self.audio_projection(a)

    @torch.inference_mode()
    def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        if a.shape[1] % int(seq_len * self.samples_per_frame) != 0:
            a = F.pad(a, (0, int(seq_len * self.samples_per_frame) - a.shape[1]), mode='replicate')
        a = self.get_wav2vec2_feature(a, seq_len=seq_len)
        return self.audio_projection(a)



######## Main Model ########
class AvatarForcing(BaseDiffusionModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.rank = opt.rank
        
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.samples_per_frame = int(self.sampling_rate / self.fps)
        self.num_frames_for_clip = int(self.opt.sec * self.fps)

        # motion autoencoder
        self.motion_autoencoder = MotionAutoencoder(size = opt.input_size, style_dim = opt.dim_w, motion_dim = opt.dim_m)
        self.motion_autoencoder.requires_grad_(False)
        self.num_frames_per_block = opt.num_frames_per_block

        # condition encoders
        self.audio_encoder = AudioEncoder(opt)

        # Flow Models
        self.num_train_timestep = opt.num_train_timestep
        self.min_step = int(0 * self.num_train_timestep)
        self.max_step = int(1 * self.num_train_timestep)
        self.flow_transformer = FlowTransformer(opt)

        self.scheduler.timesteps = self.scheduler.timesteps.to(self.rank)


    ############### Set model for training ##############
    def set_requires_grad(self, requires_grad: bool) -> None:
        for name, param in self.named_parameters():
            if "motion_autoencoder" in name or 'wav2vec2' in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(requires_grad)

    def get_parameters_for_train(self) -> list:
        return [v for k, v in self.named_parameters() if v.requires_grad]

    #####################################################
    
    @torch.inference_mode()
    def encode_user_motion(self, user_frame_list):
        outputs = []
        for i in range(0, len(user_frame_list), self.num_frames_for_clip):
            batch_frames = user_frame_list[i:i + self.num_frames_for_clip]
            user_frame_tensor = torch.cat(batch_frames, dim=0).to(self.rank)
            user_r_d = self.motion_autoencoder.enc.enc_motion(user_frame_tensor)
            user_r_d = self.motion_autoencoder.dec.direction(user_r_d)
            outputs.append(user_r_d)
        user_r_d = torch.cat(outputs, dim=0).unsqueeze(0)
        return user_r_d


    @torch.inference_mode()
    def decode_latent_into_image(
        self, s_r: torch.Tensor, r_s: torch.Tensor,
        s_r_feats: list, r_d: torch.Tensor
    ) -> dict:

        T = r_d.shape[1]
        B = r_d.shape[0]
        block_size = self.num_frames_per_block
        d_hat_blocks = []
        s_r = s_r.unsqueeze(1)
        
        s_r_feats_expanded = [f.repeat_interleave(block_size, dim=0) for f in s_r_feats]
        
        for block_idx in range(0, T, block_size):
            _t_dec = _tic()
            block_end = min(block_idx + block_size, T)
            block_len = block_end - block_idx

            r_d_block = r_d[:, block_idx:block_end]  # [B, block_len, D]
            
            needs_padding = block_len < block_size
            if needs_padding:
                pad_len = block_size - block_len
                r_d_block = F.pad(r_d_block, (0, 0, 0, pad_len), mode='replicate')
            
            s_r_d_block = s_r + r_d_block
            s_r_d_block = s_r_d_block.reshape(B * block_size, -1)

            feats_for_block = s_r_feats_expanded
            img_block, _ = self.motion_autoencoder.dec(s_r_d_block, alpha=None, feats=feats_for_block)            
            img_block = img_block.reshape(B, block_size, *img_block.shape[1:])
            
            if needs_padding: img_block = img_block[:, :block_len]
            _toc("c) decode_latent_into_image / block", _t_dec)
            d_hat_blocks.append(img_block)

        d_hat = torch.cat(d_hat_blocks, dim=1).squeeze()

        return {'d_hat': d_hat}


    #####################################################

    @torch.inference_mode()
    def update_kv_cache(
        self,
        final_latents: torch.Tensor,

        precomputed_c,
        precomputed_wr,
        precomputed_adaLN,  # not used — KV cache update always uses t=0

        start_pos: int = 0
    ):
        x_cat = torch.cat([final_latents, final_latents, final_latents], dim=0)
        B, L = x_cat.shape[0:2]
        t_zero = torch.zeros(B, L, dtype=torch.float32, device=final_latents.device)
        t_zero_embed = self.flow_transformer.t_embedder(t_zero.flatten()).unflatten(0, t_zero.shape)

        shared_scale_shift_alpha = self.flow_transformer.adaLN_modulation(t_zero_embed).chunk(6, dim=-1)
        final_adaLN = self.flow_transformer.final_layer.adaLN_modulation(t_zero_embed).chunk(2, dim=-1)
        adaLN_t0 = (shared_scale_shift_alpha, t_zero_embed, final_adaLN)

        self.flow_transformer.forward_with_precomputed(
            x                 = x_cat,
            precomputed_c     = precomputed_c,
            precomputed_adaLN = adaLN_t0,
            precomputed_wr    = precomputed_wr,
            start_pos         = start_pos,
            kv_cache_list     = self.kv_cache,
            update_kv_cache   = True)


    def initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize KV cache with shape (B*3, H, L, D) for batched CFG."""
        self.cache_len = self.num_frames_for_clip - self.num_frames_per_block
        self.cross_cache_len = self.opt.attention_window  # 2
        kv_cache = []
        head_dim = self.opt.dim_h // self.opt.num_heads
        for _ in range(self.opt.transformer_depth):
            self_kv_cache_dict = {
                "k": torch.zeros((batch_size * 3, self.opt.num_heads, self.cache_len - 2, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size * 3, self.opt.num_heads, self.cache_len - 2, head_dim), dtype=dtype, device=device)}
            cross_kv_cache_dict = {
                "k": torch.zeros((batch_size * 3, self.opt.num_heads, self.cross_cache_len, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size * 3, self.opt.num_heads, self.cross_cache_len, head_dim), dtype=dtype, device=device)}
            kv_cache.append((self_kv_cache_dict, cross_kv_cache_dict))
        self.kv_cache = kv_cache # list


    @torch.inference_mode()
    def inference(
        self,
        data: dict,
        a_cfg_scale = None,
        u_cfg_scale = None,
        nfe         = 10,
        seed        = None,
        use_kv_cache: bool = True,
    ) -> dict:
        
        s = data['avatar_ref']
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))
        if 's_r' in data:
            r_s = self.encode_identity_into_motion(s_r)
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data['r_s'] = r_s

        sample = self.sample(
            data          = data,
            a_cfg_scale   = a_cfg_scale,
            u_cfg_scale   = u_cfg_scale,
            nfe           = nfe,
            seed          = seed,
            use_kv_cache  = use_kv_cache,
        )

        data_out = self.decode_latent_into_image(
            s_r       = s_r,
            r_s       = r_s,
            s_r_feats = s_r_feats,
            r_d       = sample
        )

        return data_out

    @torch.inference_mode()
    def sample(
        self,
        data: dict         = {},
        a_cfg_scale: float = 1.0,
        u_cfg_scale: float = 1.0,
        nfe: int           = 10,
        seed: int          = None,
        use_kv_cache: bool = True
    ) -> torch.Tensor:
        r_s, avatar_a, user_a, user_frame = data['r_s'], data['avatar_a'], data['user_a'], data["user_frame"]

        B = r_s.shape[0]
        T = math.ceil(max(avatar_a.shape[-1], user_a.shape[-1]) * self.fps / self.sampling_rate)

        # Basic setups
        self.denoising_step_list = torch.tensor(np.linspace(self.opt.num_train_timestep, 0, nfe - 1).tolist())
        self.initialize_kv_cache(batch_size=B, dtype=avatar_a.dtype, device=self.rank)

        avatar_a, user_a = avatar_a.to(self.rank), user_a.to(self.rank)
        _t = _tic()
        avatar_wa = self.audio_encoder.inference(avatar_a, seq_len=T)
        user_wa  = self.audio_encoder.inference(user_a, seq_len=T)
        _toc("audio_encode(full utterance, x2)", _t)

        # Computing the first block
        avatar_wa_t = avatar_wa[:, :self.num_frames_for_clip]
        if avatar_wa_t.shape[1] < self.num_frames_for_clip:
            avatar_wa_t = F.pad(avatar_wa_t, (0, 0, 0, self.num_frames_for_clip - avatar_wa_t.shape[1]), mode='replicate')

        user_wa_t = user_wa[:, :self.num_frames_for_clip]
        if user_wa_t.shape[1] < self.num_frames_for_clip:
            user_wa_t = F.pad(user_wa_t, (0, 0, 0, self.num_frames_for_clip - user_wa_t.shape[1]), mode='replicate')

        _t = _tic()
        user_r_d = self.encode_user_motion(user_frame)
        _toc("user_motion_encode(all frames)", _t)

        user_r_d_t = user_r_d[:, :self.num_frames_for_clip]

        if user_r_d_t.shape[1] < self.num_frames_for_clip:
            user_r_d_t = F.pad(user_r_d_t, (0, 0, 0, self.num_frames_for_clip - user_r_d_t.shape[1]), mode='replicate')        
        x_t = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.rank)
        precomputed_c, precomputed_wr, precomputed_adaLN = self.prepare_cfg_condition(
            avatar_wa_t, user_wa_t, user_r_d_t, r_s, seq_len=self.num_frames_for_clip, context_len=0)

        samples = []
        start_pos = 0

        _t_prime = _tic()
        for index, current_timestep in enumerate(self.denoising_step_list):
            is_final_step = (index == len(self.denoising_step_list) - 1)
            x_t = self.solve_cfg(
                B                = B,
                index            = index,
                current_timestep = current_timestep,
                x_t              = x_t,

                precomputed_c     = precomputed_c,
                precomputed_wr    = precomputed_wr,
                precomputed_adaLN = precomputed_adaLN,

                start_pos       = start_pos,
                context_len     = 0,

                use_kv_cache    = False,
                a_cfg_scale     = a_cfg_scale,
                u_cfg_scale     = u_cfg_scale,
            )

            if is_final_step:        
                self.update_kv_cache(
                    final_latents     = x_t,
                    precomputed_c     = precomputed_c,
                    precomputed_wr    = precomputed_wr,
                    precomputed_adaLN = precomputed_adaLN,
                    start_pos         = start_pos
                )

        _toc("a) priming block denoise (50f)", _t_prime)
        samples.append(x_t)

        # Computing the subsequent blocks
        if T - self.num_frames_for_clip > 0:
            for t in range(self.num_frames_for_clip, T, self.num_frames_per_block):
                _t_block = _tic()
                start_pos = t - 2 if use_kv_cache else t - (self.num_frames_for_clip - self.num_frames_per_block)

                ss_idx, ee_idx = t, t + self.num_frames_per_block
                s_idx, e_idx   = ss_idx * self.samples_per_frame, ee_idx * self.samples_per_frame

                avatar_wa_t = avatar_wa[:, ss_idx - 2: ee_idx]
                if avatar_wa_t.shape[1] < self.num_frames_per_block + 2:
                    avatar_wa_t = F.pad(avatar_wa_t, (0, 0, 0, self.num_frames_per_block + 2 - avatar_wa_t.shape[1]), mode='replicate')

                user_wa_t = user_wa[:, ss_idx - 2: ee_idx]
                if user_wa_t.shape[1] < self.num_frames_per_block + 2:
                    user_wa_t = F.pad(user_wa_t, (0, 0, 0, self.num_frames_per_block + 2 - user_wa_t.shape[1]), mode='replicate')

                user_r_d_t = user_r_d[:, ss_idx-2: ee_idx].to(self.rank)
                if user_r_d_t.shape[1] < self.num_frames_per_block + 2:
                    user_r_d_t = F.pad(user_r_d_t, (0, 0, 0, self.num_frames_per_block + 2 - user_r_d_t.shape[1]), mode='replicate')

                # initialize noise block
                offset_x_t = x_t[:, -2:]
                noise_t = torch.randn(B, self.num_frames_per_block, self.opt.dim_w, device=self.rank)
                x_t = torch.cat([offset_x_t, noise_t], dim=1)

                _t_cond = _tic()
                precomputed_c, precomputed_wr, precomputed_adaLN = self.prepare_cfg_condition(
                    avatar_wa_t, user_wa_t, user_r_d_t, r_s, seq_len = self.num_frames_per_block, context_len = 2)
                _toc("b1) block prepare_cfg_condition", _t_cond)

                _t_denoise = _tic()
                for index, current_timestep in enumerate(self.denoising_step_list):
                    is_final_step = (index == len(self.denoising_step_list) - 1)

                    x_t = self.solve_cfg(
                        B                = B,
                        index            = index,
                        current_timestep = current_timestep,
                        x_t              = x_t,

                        precomputed_c     = precomputed_c,
                        precomputed_wr    = precomputed_wr,
                        precomputed_adaLN = precomputed_adaLN,

                        start_pos       = start_pos,
                        context_len     = 2,
                        use_kv_cache    = use_kv_cache,

                        a_cfg_scale     = a_cfg_scale,
                        u_cfg_scale     = u_cfg_scale
                    )
                    
                    if is_final_step:
                        self.update_kv_cache(
                            final_latents     = x_t,

                            precomputed_c     = precomputed_c,
                            precomputed_wr    = precomputed_wr,
                            precomputed_adaLN = precomputed_adaLN,

                            start_pos         = start_pos
                        )

                _toc("b2) block denoise loop (10f, nfe)", _t_denoise)
                _toc("b3) block total (cond+denoise)", _t_block)
                samples.append(x_t[:, -self.num_frames_per_block:])
        
        samples = torch.cat(samples, dim=1)[:, :T]
        return samples


    def solve_cfg(self, B: int, index: int, current_timestep: float, x_t: torch.Tensor,
        precomputed_c: torch.Tensor, precomputed_wr: torch.Tensor, precomputed_adaLN: torch.Tensor,
        start_pos: int, context_len: int, use_kv_cache: bool, a_cfg_scale: float, u_cfg_scale: float):

        seq_len = self.num_frames_per_block if use_kv_cache else self.num_frames_for_clip
        x_cat = x_t.repeat(3, 1, 1)
        precomputed_adaLN = precomputed_adaLN[index] if precomputed_adaLN is not None else None                                                                                                                                                                                             

        v = self.flow_transformer.forward_with_precomputed(
            x                       = x_cat, 
            precomputed_c           = precomputed_c,
            precomputed_adaLN       = precomputed_adaLN,
            precomputed_wr          = precomputed_wr,

            start_pos               = start_pos,
            kv_cache_list           = self.kv_cache if use_kv_cache else None,
            update_kv_cache         = False)

        v_uncond, v_aud, v_user = v.chunk(3, dim=0)
        v_t = v_uncond + a_cfg_scale * (v_aud - v_uncond) + u_cfg_scale * (v_user - v_uncond)

        next_timestep = self.denoising_step_list[index + 1] if index < len(self.denoising_step_list) - 1 else current_timestep

        if use_kv_cache: # kv inference
            lead, tail = 2, self.num_frames_per_block
        elif start_pos == 0: # first block (both kv and non-kv)
            lead, tail = 0, self.num_frames_for_clip
        else: # non-kv infernece
            lead, tail = self.num_frames_for_clip - self.num_frames_per_block, self.num_frames_per_block
        t_for_denoise_single = self._timestep_row(B, lead, tail, current_timestep)
        t_for_next_denoise = self._timestep_row(B, lead, tail, next_timestep)
        return self.solve_closed(flow_pred=v_t, xt=x_t, timestep=t_for_denoise_single, next_timestep=t_for_next_denoise)
        

    def prepare_cfg_condition(self, wa, wa_user, motion_user, wr, seq_len, context_len=0) -> tuple:
        B = wa.shape[0]
        null_wa = torch.zeros_like(wa)
        null_user_wa = torch.zeros_like(wa_user)
        null_user_motion = torch.zeros_like(motion_user)

        wa_cat = torch.cat([null_wa, wa, null_wa], dim=0)
        wa_user_cat = torch.cat([null_user_wa, null_user_wa, wa_user], dim=0)
        motion_user_cat = torch.cat([null_user_motion, null_user_motion, motion_user], dim=0)
        wr_cat = torch.cat([wr, wr, wr], dim=0)
        
        precomputed_c = self.flow_transformer.compute_condition(wa = wa_cat, wa_user = wa_user_cat, motion_user = motion_user_cat, train = False)
        precomputed_wr = self.flow_transformer.compute_wr_embedded(wr = wr_cat, seq_len = seq_len + context_len, train = False)

        timestep_list = [t.item() for t in self.denoising_step_list]
        precomputed_adaLN = self.flow_transformer.precompute_timestep_adaLN(timestep_list=timestep_list, batch_size=B * 3, seq_len=seq_len, context_len=context_len, device=self.rank)
        return precomputed_c, precomputed_wr, precomputed_adaLN


    def _timestep_row(self, B: int, lead: int, tail: int, value) -> torch.Tensor:
        """``[B, lead+tail]``: ``lead`` zeros (clean history) then ``value``.

        Built straight on the GPU. This is rebuilt for every denoise step -- 9x
        per 400 ms block, twice each -- and assembling it on the host cost an
        H2D copy every time.
        """
        row = torch.zeros((B, lead + tail), dtype=torch.float32, device=self.rank)
        if tail:
            # `denoising_step_list` is a CPU tensor, so float() is free here.
            row[:, lead:] = float(value)
        return row

    def _scheduler_tables(self, device):
        """The scheduler's timestep/sigma tables, cached on the GPU.

        Both are constant through inference but were copied from the host on
        every denoise step.
        """
        cache = getattr(self, "_sched_tbl_cache", None)
        if cache is None or cache[0] != device:
            cache = (device,
                     self.scheduler.timesteps.to(device=device, dtype=torch.double),
                     self.scheduler.sigmas.to(device=device, dtype=torch.double))
            self._sched_tbl_cache = cache
        return cache[1], cache[2]

    def solve_closed(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, next_timestep: torch.Tensor) -> torch.Tensor:
        orig_dtype, device = xt.dtype, xt.device

        v  = flow_pred.to(device=device, dtype=torch.double)  # [B,T,D]
        x  = xt.to(device=device, dtype=torch.double)         # [B,T,D]
        timesteps_tbl, sigmas_tbl = self._scheduler_tables(device)  # (K,), (K,)

        t_cur  = timestep.to(device=device, dtype=torch.double)         # [B,T]
        t_next = next_timestep.to(device=device, dtype=torch.double)    # [B,T]

        diffs_t  = (timesteps_tbl.view(1,1,-1) - t_cur.unsqueeze(-1)).abs()   # [B,T,K]
        diffs_tp = (timesteps_tbl.view(1,1,-1) - t_next.unsqueeze(-1)).abs()  # [B,T,K]
        idx_t    = diffs_t.argmin(dim=-1)                                   # [B,T]
        idx_tp   = diffs_tp.argmin(dim=-1)                                  # [B,T]
        
        sigma_t  = sigmas_tbl[idx_t]     # [B,T]
        sigma_tp = sigmas_tbl[idx_tp]    # [B,T]

        zero_t  = (timestep == 0).to(device=device)
        zero_tp = (next_timestep == 0).to(device=device)
        sigma_t  = torch.where(zero_t,  torch.zeros_like(sigma_t),  sigma_t)
        sigma_tp = torch.where(zero_tp, torch.zeros_like(sigma_tp), sigma_tp)

        alpha_t, alpha_tp = 1.0 - sigma_t, 1.0 - sigma_tp

        x0_hat = x - sigma_t.unsqueeze(-1) * v   # [B,T,D]
        # `where`, not `ratio[valid] = ...`: boolean-mask indexing has a
        # data-dependent output shape, so it synchronises the CUDA stream --
        # three times per denoise step, 27 times per 400 ms block. The clamp
        # keeps the discarded branch finite, so the two forms agree exactly.
        valid = (sigma_t > 0)
        ratio = torch.where(valid, sigma_tp / torch.clamp(sigma_t, min=1e-8),
                            torch.zeros_like(sigma_tp))
        xt_next = (ratio.unsqueeze(-1) * x + (alpha_tp - ratio * alpha_t).unsqueeze(-1) * x0_hat)
        hist_mask = (zero_t & zero_tp).unsqueeze(-1)  # [B,T,1]
        xt_next = torch.where(hist_mask, x, xt_next)
        return xt_next.to(orig_dtype)


    ######## Motion encoder - decoder ########
    @torch.inference_mode()
    def encode_image_into_latent(self, x: torch.Tensor) -> list:
        x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats


    @torch.inference_mode()
    def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
        if len(x_r.shape) == 3:
            b, t = x_r.shape[0:2]
            x_r = x_r.reshape(b*t, *x_r.shape[2:])
            x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
            r_x = self.motion_autoencoder.dec.direction(x_r_lambda).reshape(b, t, -1)
        else:
            x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
            r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        return r_x


    @torch.inference_mode()
    def encode_identity_into_lambda(self, x_r: torch.Tensor) -> torch.Tensor:
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r_lambda

    @torch.inference_mode()
    def decode_block(
        self,
        r_d_block: torch.Tensor,
        s_r: torch.Tensor,
        s_r_feats_expanded: list,
        block_size: int,
        B: int
    ) -> torch.Tensor:
        block_len = r_d_block.shape[1]
        
        s_r_d_block = s_r + r_d_block
        s_r_d_block = s_r_d_block.reshape(B * block_len, -1)

        img_block, _ = self.motion_autoencoder.dec(s_r_d_block, alpha=None, feats=s_r_feats_expanded)
        img_block = img_block.reshape(B, block_len, *img_block.shape[1:])
        return img_block.squeeze(0)

