"""Evaluate room detection and fall-risk zones against hand-labelled real-building maps.

    cd server && ../.venv/bin/python -m sightline.evaluate            # tune on area A, score on area B
    cd server && ../.venv/bin/python -m sightline.evaluate --spare    # also score the spare area C (do this once, at the end)

Method (honest by construction):
  1. Data: real walks of the building, saved as maps (commander page -> Save map). Each saved map is one area
     and is labelled by hand on the 2D map (Label mode): its rooms (polygons) and fall-risk spots (points).
  2. Split by area, never by frame: area A = training, area B = test, area C = spare (held out).
  3. Training: a small grid search over the detectors' settings (rooms.PARAMS, risk.PARAMS) on area A only.
  4. Test: the chosen settings are scored once on area B. Area C is only scored with --spare.
  5. Metrics: rooms matched one-to-one at IoU >= 0.5 -> precision, recall, mean best IoU;
     fall-risk: a labelled spot is found if it lies in (or within 30 cm of) a predicted zone -> recall;
     a predicted zone is correct if a labelled spot lies in it -> precision; F1 from both.
  6. Everything is written to evaluation/report-<date>.md with the data used; the tuned settings to
     data/tuned.json, which the live system then uses. Nothing is estimated or filled in by hand.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from . import risk as RK
from . import rooms as RM
from .maps import MapStore, restore
from .records import Recorder
from .security import DATA

TOL = 0.3  # m: a labelled risk spot within this of a predicted zone counts as found
REPORTS = Path(__file__).resolve().parents[2] / "evaluation"
ROOM_GRID = {"min_area": [1.0, 2.0, 3.0, 4.0], "wall_thick": [0, 1, 2]}
RISK_GRID = {"stairs_margin": [0.3, 0.6, 0.9], "clutter_max_top": [0.25, 0.35, 0.45], "clutter_margin": [0.2, 0.35, 0.5],
             "drop_radius": [0.3, 0.5, 0.8], "fall_radius": [0.6, 1.0, 1.4]}


def load_tuned() -> None:
    """Apply settings tuned by a previous evaluation (data/tuned.json) to the live detectors."""
    f = DATA / "tuned.json"
    if f.exists():
        t = json.loads(f.read_text())
        RM.PARAMS.update(t.get("rooms", {}))
        RK.PARAMS.update(t.get("risk", {}))


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def score_rooms(pred: list[dict], labels: list[list[list[float]]]) -> dict:
    if not labels:
        return {"n_label": 0, "n_pred": len(pred), "precision": None, "recall": None, "f1": None, "mean_iou": None}
    pts = [p for poly in labels for p in poly] + [c for r in pred for c in r["poly"]]
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    gx = np.arange(min(xs), max(xs), 0.1) + 0.05
    gz = np.arange(min(zs), max(zs), 0.1) + 0.05
    grid = [(x, z) for x in gx for z in gz]
    lab_masks = [np.array([RK.inside({"poly": poly}, x, z) for x, z in grid]) for poly in labels]
    pred_masks = [np.array([RM.contains(r, x, z) for x, z in grid]) for r in pred]
    iou = np.zeros((len(labels), len(pred)))
    for i, a in enumerate(lab_masks):
        for j, b in enumerate(pred_masks):
            union = (a | b).sum()
            iou[i, j] = (a & b).sum() / union if union else 0.0
    matched, used = 0, set()
    for i in np.argsort(-iou.max(axis=1) if len(pred) else np.zeros(len(labels))):
        if not len(pred):
            break
        j = int(np.argmax([iou[i, k] if k not in used else -1 for k in range(len(pred))]))
        if j not in used and iou[i, j] >= 0.5:
            used.add(j)
            matched += 1
    p = matched / len(pred) if pred else 0.0
    r = matched / len(labels)
    return {"n_label": len(labels), "n_pred": len(pred), "precision": round(p, 3), "recall": round(r, 3), "f1": round(_f1(p, r), 3),
            "mean_iou": round(float(iou.max(axis=1).mean()) if len(pred) else 0.0, 3)}


def score_risk(zones: list[dict], spots: list[list[float]]) -> dict:
    if not spots:
        return {"n_label": 0, "n_pred": len(zones), "precision": None, "recall": None, "f1": None}
    found = sum(1 for x, z in spots if any(RK.inside(zn, x, z, TOL) for zn in zones))
    correct = sum(1 for zn in zones if any(RK.inside(zn, x, z, TOL) for x, z in spots))
    p = correct / len(zones) if zones else 0.0
    r = found / len(spots)
    return {"n_label": len(spots), "n_pred": len(zones), "precision": round(p, 3), "recall": round(r, 3), "f1": round(_f1(p, r), 3)}


def _pool(scores: list[dict]) -> dict:
    """Micro-average over maps: sum the counts, not the ratios."""
    s = [x for x in scores if x["n_label"]]
    if not s:
        return {"maps": 0}
    lab = sum(x["n_label"] for x in s)
    pred = sum(x["n_pred"] for x in s)
    tp_r = sum(x["recall"] * x["n_label"] for x in s)
    tp_p = sum(x["precision"] * x["n_pred"] for x in s)
    p, r = (tp_p / pred if pred else 0.0), tp_r / lab
    out = {"maps": len(s), "labels": lab, "predicted": pred, "precision": round(p, 3), "recall": round(r, 3), "f1": round(_f1(p, r), 3)}
    if "mean_iou" in s[0]:
        out["mean_iou"] = round(sum(x["mean_iou"] * x["n_label"] for x in s) / lab, 3)
    return out


def _predict(world, room_p: dict, risk_p: dict) -> tuple[list[dict], list[dict]]:
    walls = world.merged_walls()
    return RM.segment_rooms(walls, room_p), RK.fall_risk_zones(world.stairs, world.drops, world.objects, walls, world.fall_spots, risk_p)


def evaluate(items: list, room_p: dict, risk_p: dict) -> tuple[dict, dict]:
    rs, ks = [], []
    for world, labels in items:
        rooms, zones = _predict(world, room_p, risk_p)
        rs.append(score_rooms(rooms, labels.get("rooms", [])))
        ks.append(score_risk(zones, labels.get("risk", [])))
    return _pool(rs), _pool(ks)


def tune(items: list) -> tuple[dict, dict]:
    # start from the defaults and only move to settings that score strictly better on the training area
    best_room = dict(RM.PARAMS)
    best_rf = evaluate(items, best_room, RK.PARAMS)[0].get("f1", 0) or 0
    for combo in itertools.product(*ROOM_GRID.values()):
        p = {**RM.PARAMS, **dict(zip(ROOM_GRID, combo))}
        f = evaluate(items, p, RK.PARAMS)[0].get("f1", 0) or 0
        if f > best_rf + 1e-9:
            best_room, best_rf = p, f
    best_risk = dict(RK.PARAMS)
    best_kf = evaluate(items, RM.PARAMS, best_risk)[1].get("f1", 0) or 0
    for combo in itertools.product(*RISK_GRID.values()):
        p = {**RK.PARAMS, **dict(zip(RISK_GRID, combo))}
        f = evaluate(items, RM.PARAMS, p)[1].get("f1", 0) or 0
        if f > best_kf + 1e-9:
            best_risk, best_kf = p, f
    return best_room, best_risk


def main() -> None:
    from .state import World

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spare", action="store_true", help="also score the held-out area C (use once, at the very end)")
    a = ap.parse_args()
    store = MapStore(Recorder().fernet)
    by_area: dict[str, list] = {"A": [], "B": [], "C": []}
    used = []
    for m in store.list():
        labels = store.labels(m["id"])
        if not labels or labels.get("area") not in by_area:
            continue
        w = World()
        restore(w, store.load(m["id"]))
        by_area[labels["area"]].append((w, labels))
        used.append((m["name"], labels["area"], len(labels.get("rooms", [])), len(labels.get("risk", []))))
    if not by_area["A"] or not by_area["B"]:
        raise SystemExit("Need labelled maps for area A (training) and area B (test): commander page -> Saved maps -> Label.")
    default_room, default_risk = dict(RM.PARAMS), dict(RK.PARAMS)
    room_p, risk_p = tune(by_area["A"])
    results = {
        "train (A), default settings": evaluate(by_area["A"], default_room, default_risk),
        "train (A), tuned settings": evaluate(by_area["A"], room_p, risk_p),
        "test (B), default settings": evaluate(by_area["B"], default_room, default_risk),
        "test (B), tuned settings": evaluate(by_area["B"], room_p, risk_p),
    }
    if a.spare and by_area["C"]:
        results["spare (C), tuned settings"] = evaluate(by_area["C"], room_p, risk_p)
    (DATA / "tuned.json").write_text(json.dumps({"rooms": room_p, "risk": risk_p, "tuned_on": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"report-{time.strftime('%Y%m%d-%H%M')}.md"
    lines = [f"# FindMyTeam detector evaluation ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "Generated by `python -m sightline.evaluate` from hand-labelled maps of real walks. Method: see the module docstring.", "",
             "## Data", "", "| saved map | area | labelled rooms | labelled fall-risk spots |", "|---|---|---|---|"]
    lines += [f"| {n} | {ar} | {r} | {k} |" for n, ar, r, k in used]
    lines += ["", "Area A = training (settings tuned here only), B = test (scored once), C = spare (held out"
              + (", scored in this run)." if a.spare else ", not scored in this run)."), "",
              "## Rooms (matched at IoU >= 0.5)", "", "| run | maps | labelled | predicted | precision | recall | F1 | mean IoU |", "|---|---|---|---|---|---|---|---|"]
    for k, (rm, _) in results.items():
        lines.append(f"| {k} | {rm.get('maps', 0)} | {rm.get('labels', '-')} | {rm.get('predicted', '-')} | {rm.get('precision', '-')} | "
                     f"{rm.get('recall', '-')} | {rm.get('f1', '-')} | {rm.get('mean_iou', '-')} |")
    lines += ["", f"## Fall-risk zones (spot found within {TOL:.1f} m)", "", "| run | maps | labelled | predicted | precision | recall | F1 |", "|---|---|---|---|---|---|---|"]
    for k, (_, rk) in results.items():
        lines.append(f"| {k} | {rk.get('maps', 0)} | {rk.get('labels', '-')} | {rk.get('predicted', '-')} | {rk.get('precision', '-')} | "
                     f"{rk.get('recall', '-')} | {rk.get('f1', '-')} |")
    lines += ["", "## Settings", "", f"- Rooms default: `{default_room}`", f"- Rooms tuned on A: `{room_p}`",
              f"- Fall-risk default: `{default_risk}`", f"- Fall-risk tuned on A: `{risk_p}`", "",
              "Tuned settings saved to `server/data/tuned.json`; the live system loads them at start-up."]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
