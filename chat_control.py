"""
Natural-language control over the render adjustments, via a local Ollama
model (qwen3:8b) using tool-calling: the user's free text ("add in pedal",
"play with more rubato") gets mapped to a structured update of the
adjustment knobs in infer.py, rather than parsed with brittle keyword rules.
"""
import json

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

DEFAULT_PARAMS = {
    "era": "romantic",
    "rubato_intensity": 1.0,      # 0 = robotic/no timing deviation, 1 = model's normal prediction, >1 = exaggerated
    "dynamics_intensity": 1.0,    # 0 = flat velocity, 1 = normal, >1 = exaggerated dynamic range
    "pedal_scale": 1.0,           # multiplier on predicted pedal amount, 0 = none, >1 = heavier
    "pedal_boost": 0.0,           # flat additive pedal, -1..1, use positive to add pedal even if near zero
    "tempo_multiplier": 1.0,      # overall playback speed, 1 = normal, >1 = faster, <1 = slower
    "articulation_intensity": 1.0,  # 0 = no staccato/legato variation, 1 = normal, >1 = exaggerated
}

ADJUST_TOOL = {
    "type": "function",
    "function": {
        "name": "adjust_render",
        "description": (
            "Adjust the expressive-performance rendering parameters based on the "
            "user's request. Only include fields that should change; omit anything "
            "the user didn't ask about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "era": {
                    "type": "string",
                    "enum": ["baroque", "classical", "romantic", "modern"],
                    "description": "musical style era to render in",
                },
                "rubato_intensity": {
                    "type": "number",
                    "description": "0=robotic/no timing deviation, 1=normal, 2=very exaggerated rubato",
                },
                "dynamics_intensity": {
                    "type": "number",
                    "description": "0=flat/constant velocity, 1=normal, 2=very exaggerated dynamics",
                },
                "pedal_scale": {
                    "type": "number",
                    "description": "multiplier on predicted sustain pedal amount: 0=none, 1=normal, 2=heavy",
                },
                "pedal_boost": {
                    "type": "number",
                    "description": (
                        "flat additive pedal amount from -1 to 1; use a positive value "
                        "to add pedal even when the predicted amount is currently near zero"
                    ),
                },
                "tempo_multiplier": {
                    "type": "number",
                    "description": "overall playback speed: 1=normal, >1=faster, <1=slower",
                },
                "articulation_intensity": {
                    "type": "number",
                    "description": "0=no staccato/legato variation, 1=normal, 2=very exaggerated",
                },
                "reply": {
                    "type": "string",
                    "description": "a short, friendly confirmation of what changed, shown to the user",
                },
            },
            "required": ["reply"],
        },
    },
}

SYSTEM_PROMPT = """You control an expressive piano performance renderer through a tool call.
The user describes how they want the performance to sound (e.g. "add in pedal",
"play with more rubato", "make it sound baroque", "less dynamic, more even").
Call adjust_render with only the parameters implied by their request, as NEW
absolute values (not deltas) - you are told the current values below, so reason
about what a sensible new absolute value is given what they're asking for.
Always include a short "reply" confirming what you changed, in plain friendly
language, no more than one sentence."""


def interpret_command(message, current_params):
    """Returns (updated_params, reply_text)."""
    params_desc = ", ".join(f"{k}={v}" for k, v in current_params.items())
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCurrent values: {params_desc}"},
        {"role": "user", "content": message},
    ]

    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "tools": [ADJUST_TOOL],
        "stream": False,
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    tool_calls = data.get("message", {}).get("tool_calls") or []
    if not tool_calls:
        fallback = data.get("message", {}).get("content") or "I didn't catch an adjustment in that - try something like 'add more pedal'."
        return dict(current_params), fallback

    args = tool_calls[0]["function"]["arguments"]
    if isinstance(args, str):
        args = json.loads(args)

    reply = args.pop("reply", "Done.")
    new_params = dict(current_params)
    for k, v in args.items():
        if k in new_params:
            new_params[k] = v
    return new_params, reply
