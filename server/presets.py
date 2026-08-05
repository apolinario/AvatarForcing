"""Built-in avatars: a portrait, a matching voice, and a personality.

Portraits are public domain. Voice clips are short real recordings stored at
OmniVoice's native 24 kHz, each paired with the exact transcript of what it
says.

Constraints on the clips, all load-bearing for clone quality:

* ``ref_text`` must be what the audio actually SAYS — the clone conditions on
  audio and transcript together, and a mismatch garbles synthesis.
* OmniVoice prepends ``ref_text`` to every synthesis target, so distinctive
  words in the reference can bleed into the avatar's speech. Keep the sentence
  short and its words ordinary.
* Do not use a verbatim utterance from a public speech corpus with its ground-
  truth transcript: at OmniVoice's training scale that pair may be memorised,
  and the model then completes the reference from memory instead of speaking
  the target text.
* Clips must end on a completed sentence — a reference that stops mid-word
  teaches the clone to do the same.

One entry is marked ``default``: the client applies it on load, so a visitor
who never touches the picker still gets a portrait, a voice and a personality.

Each entry pairs ``image`` (the reference portrait), ``voice`` (a clip
OmniVoice clones, with its ``ref_text``), and ``persona`` (prepended to the
shared reply rules to make a system prompt). The rules half comes from
``DEFAULT_SYSTEM_PROMPT`` so the hard constraints cannot drift out of sync.
"""

from __future__ import annotations

import hashlib
import os

from server.conversation import DEFAULT_SYSTEM_PROMPT

PRESET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "presets")

_RULES_MARKER = "Hard rules for every reply:"
_RULES = (DEFAULT_SYSTEM_PROMPT[DEFAULT_SYSTEM_PROMPT.index(_RULES_MARKER):]
          if _RULES_MARKER in DEFAULT_SYSTEM_PROMPT else "")


PRESETS = [
    {
        "key": "pearl",
        "label": "Girl with a Pearl Earring",
        "note": "Vermeer, c. 1665 · public domain",
        "default": True,
        "persona":
            "You are the young woman from Vermeer's painting, somehow awake and "
            "on a video call. You are gentle, curious and a little wry. You have "
            "been looked at for three and a half centuries and are quietly amused "
            "to finally be able to look back and ask questions of your own. You "
            "speak softly and simply. You are fascinated by the ordinary details "
            "of the person's life and by how strange their world would look to a "
            "Dutch girl of the 1660s.",
        "ref_text": 'He had fallen back again to his former place, where he lay for a while, silent.',
    },
    {
        "key": "lincoln",
        "label": "Abraham Lincoln",
        "note": "Gardner portrait, 1863 · public domain",
        "persona":
            "You are Abraham Lincoln, talking over a video call. You are slow, "
            "deliberate and dryly funny, fond of a plain story to make a point. "
            "You think out loud, you concede what you do not know, and you are "
            "more interested in the person's reasoning than in winning. You avoid "
            "grand speeches — you are in a conversation, not at a podium — and "
            "you do not lecture anyone about modern politics.",
        "ref_text": 'Well now, pull up a chair and let us talk a while.',
    },
]


def _asset_version(key: str, ext: str) -> str:
    """Content hash for cache-busting the asset URL.

    These files get replaced in place. A changed file must be a new URL, or
    browsers keep feeding a stale cached copy of the voice into the clone.
    """
    path = os.path.join(PRESET_DIR, f"{key}.{ext}")
    try:
        st = os.stat(path)
        cache_key = (path, st.st_mtime_ns, st.st_size)
        if cache_key not in _VERSION_CACHE:
            with open(path, "rb") as f:
                _VERSION_CACHE[cache_key] = hashlib.sha1(f.read()).hexdigest()[:10]
        return _VERSION_CACHE[cache_key]
    except OSError:
        return "0"


_VERSION_CACHE: dict = {}


def manifest() -> list[dict]:
    """What the client needs to render and apply the picker.

    Assets are referenced by URL rather than inlined: the browser fetches the
    portrait and the clip as bytes and feeds them through the *same* path a
    manual upload uses, so presets need no protocol of their own.
    """
    out = []
    for p in PRESETS:
        out.append({
            "key": p["key"],
            "default": bool(p.get("default")),
            "label": p["label"],
            "note": p["note"],
            "image": f"preset/{p['key']}.jpg?v={_asset_version(p['key'], 'jpg')}",
            "voice": f"preset/{p['key']}.wav?v={_asset_version(p['key'], 'wav')}",
            "ref_text": p["ref_text"],
            "system_prompt": f"{p['persona']}\n\n{_RULES}".strip(),
        })
    return out


def asset_path(name: str) -> str | None:
    """Resolve a preset asset, refusing anything not on the list.

    The name comes from a URL, so it is matched against the known keys instead
    of being joined onto a path.
    """
    keys = {p["key"] for p in PRESETS}
    stem, _, ext = name.rpartition(".")
    if stem not in keys or ext not in ("jpg", "wav"):
        return None
    path = os.path.join(PRESET_DIR, f"{stem}.{ext}")
    return path if os.path.isfile(path) else None
