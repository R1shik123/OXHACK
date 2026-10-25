"""Ask Gemini: the commander's AI assistant, answering rescuers' questions with the live situation.

Security / privacy:
  - The API key stays on the laptop: read from the GEMINI_API_KEY environment variable or from
    server/data/gemini.key (owner-only file). It is never sent to a browser, logged, or put in a URL.
  - Only the commander page (admin session) can ask. Requests are rate-limited.
  - Each question goes to Google's Gemini API together with a short text summary of the live situation
    (responder names, positions in metres from the entrance, alerts, hazards, exits, stairs, coverage,
    recent timeline). No camera images, no device tokens, no recordings.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from .security import DATA

log = logging.getLogger("sightline")
API = "https://generativelanguage.googleapis.com/v1beta"
# Tried in order until one answers: newest free-tier Flash first, then lighter / open fallbacks. Google retires
# model names (2.5 Flash is closed to new users) and free keys don't get every model, so never rely on one.
MODELS = [m for m in (os.environ.get("GEMINI_MODEL"), "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.5-flash-lite",
                      "gemini-flash-lite-latest", "gemma-4-31b-it") if m]
RETRY = {404, 429, 500, 503}  # retired, no (free) quota, overloaded: try the next model
SYSTEM = """You are the AI assistant built into FindMyTeam, a live indoor map used by an emergency-response \
incident commander. Rescuers inside the building radio questions to the commander, who types them to you.

Rules:
- Answer briefly and practically (a few short lines or bullets). The commander is busy.
- Use the LIVE SITUATION below for anything about this incident: who is where, alerts, hazards, exits, \
stairs, what has been searched. Positions are metres from the entrance marker: x to the right, z into the building.
- Never invent positions, people, routes or facts. If the situation data doesn't say, say so plainly.
- For general rescue, first-aid or safety questions, give sound general guidance and remind them to follow \
their own procedures and command decisions.
- Life-threatening emergency outside this tool: tell them to call emergency services.
- You advise; the commander decides."""


def _key() -> str | None:
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    f = DATA / "gemini.key"
    try:
        return f.read_text().strip() or None
    except OSError:
        return None


def configured() -> bool:
    return _key() is not None


def situation(world) -> str:
    """Plain-text summary of the live incident for the model (no images, no secrets)."""
    now = time.time()
    t = int(now - world.started)
    lines = [f"Incident time: T+{t // 60:02d}:{t % 60:02d}."]
    walls = world.merged_walls()
    doors = [w for w in walls if w[4] == "door"]
    pct = world.coverage.percent()
    lines.append(f"Map: {sum(1 for w in walls if w[4] == 'wall')} walls, {len(doors)} doors, {len(world.stairs)} flights of stairs, "
                 f"{'unknown' if pct is None else f'{pct:.0f}%'} of the building outline searched.")
    plan = world.plan or {}
    people = plan.get("people", {})
    lines.append("Responders:")
    if not world.phones:
        lines.append("- none connected")
    for p in world.phones.values():
        d = p.public(now)
        bits = []
        if d["pos"]:
            x, y, z = d["pos"]
            bits.append(f"at ({x:.1f}, {z:.1f}), {((x * x + z * z) ** 0.5):.1f} m from the entrance")
        else:
            bits.append("position unknown (not lined up with the marker yet)")
        if not d["connected"]:
            bits.append("OFFLINE")
        elif d["stale"]:
            bits.append("no recent update")
        if d.get("activity"):
            bits.append(d["activity"]["state"])
        info = people.get(p.id, {})
        if info.get("exit_m") is not None:
            bits.append(f"{info['exit_m']} m to the nearest exit (~{info.get('exit_s', '?')} s)")
        if info.get("turnaround_s") is not None:
            bits.append(f"turn back in {info['turnaround_s'] // 60} min")
        if d.get("battery") is not None:
            bits.append(f"battery {round(d['battery'] * 100)}%")
        if d.get("alert"):
            a = d["alert"]
            bits.append(f"ALERT: {'possible fall' if a['kind'] == 'fall' else 'not moving'} for {a['for']:.0f} s"
                        + (" (acknowledged)" if a.get("acked") else ""))
        for h in d.get("hazard") or []:
            bits.append(f"approaching hazard '{h['label']}' ({h['dist']} m, {h['why']})")
        lines.append(f"- {p.name}: " + "; ".join(bits))
    pins = list(world.pins.values())
    if pins:
        lines.append("Pins: " + "; ".join(f"{pn['kind']}{(' ' + repr(pn['label'])) if pn.get('label') else ''} at "
                                          f"({pn['pos'][0]:.1f}, {pn['pos'][2]:.1f}) by {pn.get('by', '?')}" for pn in pins))
    from .predict import exit_doors
    exits = ["entrance marker (0.0, 0.0)"] + [f"door ({x:.1f}, {z:.1f})" for x, z in exit_doors(walls)] + \
            [f"rally point ({pn['pos'][0]:.1f}, {pn['pos'][2]:.1f})" for pn in pins if pn["kind"] == "rally"]
    lines.append("Exits: " + "; ".join(exits))
    if world.stairs:
        lines.append("Stairs: " + "; ".join(
            f"({sum(c[0] for c in s['corners']) / 4:.1f}, {sum(c[1] for c in s['corners']) / 4:.1f}) going {'down' if s['down'] else 'up'} "
            f"{s['rise']:.1f} m" for s in world.stairs))
    decisions = plan.get("decisions") or []
    if decisions:
        lines.append("Planner suggestions: " + "; ".join(d["text"] for d in decisions[:6]))
    evs = list(world.events)[-15:]
    if evs:
        lines.append("Recent timeline:")
        for ev in evs:
            et = int(ev["ts"] - world.started)
            lines.append(f"- T+{et // 60:02d}:{et % 60:02d} {ev['text']}")
    return "\n".join(lines)


class Gemini:
    def __init__(self) -> None:
        self.model = MODELS[0]  # the one that answered last; tried first next time

    def _order(self) -> list[str]:
        return [self.model] + [m for m in MODELS if m != self.model]

    async def _generate(self, client: httpx.AsyncClient, key: str, body: dict) -> dict:
        """POST to the first model in the list that answers; remembers it. Raises RuntimeError with Google's reason."""
        last = ""
        for model in self._order():
            try:
                r = await client.post(f"{API}/models/{model}:generateContent", headers={"x-goog-api-key": key}, json=body)
            except httpx.TimeoutException:
                last = f"{model} timed out"
                continue
            if r.status_code == 200:
                self.model = model
                return r.json()
            msg = r.json().get("error", {}).get("message", "") if r.headers.get("content-type", "").startswith("application/json") else ""
            log.warning("gemini: %s HTTP %s %s", model, r.status_code, msg[:160])
            if r.status_code in (400, 401, 403):
                raise RuntimeError("Google rejected this key. Check it was copied completely from Google AI Studio "
                                   "(aistudio.google.com, Get API key). " + msg[:160])
            last = f"{r.status_code} {msg[:160]}"
            if r.status_code not in RETRY:
                break
        raise RuntimeError(f"No Gemini model answered right now ({last}). If this keeps happening, the key's free quota "
                           "may be used up for today.")

    async def check(self) -> tuple[bool, str]:
        """Does the saved key actually get answers? (one tiny test question). Returns (ok, message)."""
        key = _key()
        if key is None:
            return False, "No key saved."
        body = {"contents": [{"role": "user", "parts": [{"text": "Reply with OK."}]}], "generationConfig": {"maxOutputTokens": 8}}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                await self._generate(client, key, body)
            return True, f"Key works. Using {self.model}."
        except RuntimeError as e:
            return False, str(e)
        except httpx.HTTPError:
            return False, "Couldn't reach Google. Check this laptop's internet connection."

    async def ask(self, messages: list[dict], context: str) -> str:
        key = _key()
        if key is None:
            raise RuntimeError("no key")
        contents = [{"role": "model" if m["role"] == "model" else "user", "parts": [{"text": m["text"]}]} for m in messages]
        body = {"systemInstruction": {"parts": [{"text": SYSTEM + "\n\nLIVE SITUATION (right now):\n" + context}]},
                "contents": contents, "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024}}
        async with httpx.AsyncClient(timeout=40) as client:
            data = await self._generate(client, key, body)
        cands = data.get("candidates") or []
        parts = (cands[0].get("content") or {}).get("parts", []) if cands else []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if not text:
            reason = (cands[0].get("finishReason") if cands else None) or data.get("promptFeedback", {}).get("blockReason", "no answer")
            return f"(Gemini gave no answer: {reason}.)"
        return text
