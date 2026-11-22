#!/usr/bin/env python3
"""
route_solver.py - Plan a vehicle chest-breaking route that maximizes expected chests broken
within a time budget (an orienteering problem).

    python scripts/route_solver.py data/loot.csv data/minimap.png --minutes 3
    python scripts/route_solver.py data/loot.csv data/minimap.png --minutes 3 --seed-opening
    python scripts/route_solver.py data/loot.csv data/minimap.png --start -19300 -35100 --minutes 5

Model (MVP - every number lives in config/route_config.json and is meant to be replaced by measurements):
  * Value of a chest = spawn_prob of its class. Every chest rolls the same treasure odds, so maximizing
    expected chests broken = maximizing expected treasure rolls.
  * Travel = shortest path on a land/water grid derived from the minimap colors, at vehicle speed.
    Water is allowed but slow, because thin bridges vanish when the minimap is downsampled.
  * Car-reachable = class allowed by config AND chest is near local ground level (ground floor / outdoor).
  * Each chest costs break_seconds of slowdown.
Outputs: route CSV, route drawn on the minimap, and per-tier odds for the planned route.
"""
import argparse
import csv
import heapq
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEFAULT_CONFIG = {
    "_comment": "Placeholders except where marked. Replace spawn_prob values with in-game counts.",
    "vehicle_speed_mps": 17.0,          # calibrated from the user's 15 s opening (~250 m)
    "water_speed_factor": 0.25,         # water cells are 4x slower (bridges get lost in the mask)
    "break_seconds": 1.0,
    "ground_floor_tolerance_cm": 450,   # chest counts as car-reachable if within this of local ground
    "ground_search_radius_cm": 3000,
    "dedupe_radius_cm": 100,
    "water_level_cm": 300,              # chest on a water pixel below this height = on a boat
    "grid_cells": 256,
    "max_uphill_slope": 0.6,            # rise/run the van can climb (~31 deg); steeper uphill = blocked
    "elevation_percentile": 15,         # per-cell low percentile of actor Z = ground estimate
    "elevation_smooth_cells": 1.0,
    "classes": {
        "AlwaysSpawn_NormalChest_C":            {"spawn_prob": 1.0, "car": True},   # user-confirmed always spawn
        "AlwaysSpawn_RareChest_C":              {"spawn_prob": 1.0, "car": True},
        "Tiered_Chest_Athena_C":                {"spawn_prob": 0.5, "car": True},   # PLACEHOLDER
        "Tiered_Chest_5_Child_C":               {"spawn_prob": 0.5, "car": True},   # PLACEHOLDER
        "B_Athena_BuriedChest_Sand_Spawner_C":  {"spawn_prob": 0.5, "car": False},  # must be dug up
        "B_Athena_BuriedChest_Dirt_Spawner_C":  {"spawn_prob": 0.5, "car": False},
        "BGA_LootCannon_StandardChest_Pirate_C":{"spawn_prob": 1.0, "car": False},  # interact, not break
    },
    "treasure_odds_per_chest": {             # Epic-published, Nov 2026
        "Pirate Shop Access": 1 / 137500,
        "Weekly Gold Outfit": 1 / 1562500,
        "Weekly Vehicle and Decal": 1 / 7813,
        "Weekly Pickaxe or Back Bling": 1 / 568,
    },
    "treasure_hour_multiplier": {"Pirate Shop Access": 1.5},
    "minimap_calibration": {"cm_per_px": 146.0, "origin_px": 968.0, "origin_py": 1014.0, "ref_size": 2048},
    "opening": {
        "_comment": "User's known 15 s opener, as world X,Y (cm). Trailer x2 -> shack -> shack -> shrine chest.",
        "start": [-19300, -35100],
        "stops": [[-19671, -35405], [-18878, -34736], [-9918, -31227], [-7814, -27538], [-9644, -25428]],
    },
}
CONFIG_PATH = os.path.join("config", "route_config.json")


# ----------------------------------------------------------------------------- config / data
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            cfg = json.load(fh)
        for k, v in DEFAULT_CONFIG.items():   # fill any keys added in newer versions
            cfg.setdefault(k, v)
        return cfg
    os.makedirs("config", exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(DEFAULT_CONFIG, fh, indent=2)
    print(f"Wrote default config -> {CONFIG_PATH} (edit spawn_prob etc. there)")
    return json.loads(json.dumps(DEFAULT_CONFIG))


def load_chests(path, cfg, grid):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((r["class"], float(r["x"]), float(r["y"]), float(r["z"])))
            except (KeyError, ValueError):
                continue
    # ground reference = containers standing on LAND (incl. ammo). Boat chests on water are excluded,
    # otherwise a bridge over water looks like an upper floor.
    all_pts = np.array([[x, y, z] for _, x, y, z in rows if grid.on_land(x, y)])
    classes = cfg["classes"]
    # dedupe: identical placements extracted twice (seen: 24 Tiered_Chest pairs)
    kept, seen = [], {}
    rad = cfg["dedupe_radius_cm"]
    for cls, x, y, z in rows:
        if cls not in classes:
            continue
        key = (round(x / rad), round(y / rad), round(z / rad))
        if key in seen:
            continue
        seen[key] = True
        kept.append({"class": cls, "x": x, "y": y, "z": z})
    # ground-floor estimate: lowest loot container within radius approximates local ground height
    from scipy.spatial import cKDTree
    tree = cKDTree(all_pts[:, :2])
    for c in kept:
        idx = tree.query_ball_point([c["x"], c["y"]], cfg["ground_search_radius_cm"])
        ground = all_pts[idx, 2].min() if idx else c["z"]
        c["height_above_ground"] = c["z"] - ground
        c["ground_floor"] = c["height_above_ground"] <= cfg["ground_floor_tolerance_cm"]
        spec = classes[c["class"]]
        c["prize"] = spec["spawn_prob"]
        c["on_water"] = (not grid.on_land(c["x"], c["y"])) and c["z"] < cfg["water_level_cm"]
        c["car"] = bool(spec["car"]) and c["ground_floor"] and not c["on_water"]
    return kept


# ----------------------------------------------------------------------------- travel grid
class Grid:
    def __init__(self, minimap_path, cfg, elev_xyz=None):
        img = Image.open(minimap_path).convert("RGB")
        self.img = img
        self.size = img.size[0]
        cal = cfg["minimap_calibration"]
        self.k = self.size / cal["ref_size"]
        self.cal = cal
        a = np.asarray(img).astype(int)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        land = (r > b + 15) | (g > b + 25)
        land = ndimage.binary_opening(land, iterations=2)
        land = ndimage.binary_fill_holes(land)
        self.land_full = land
        n = cfg["grid_cells"]
        self.n = n
        cell_px = self.size / n
        # a cell is land if >=30% of its pixels are land
        small = np.asarray(Image.fromarray((land * 255).astype(np.uint8)).resize((n, n), Image.BOX)) / 255.0
        self.land = small >= 0.3
        self.cell_m = cell_px / self.k * cal["cm_per_px"] / 100.0
        self.elev = self.build_elevation(elev_xyz, cfg) if elev_xyz is not None else None
        v = cfg["vehicle_speed_mps"]
        cost_cell = np.where(self.land, self.cell_m / v, self.cell_m / (v * cfg["water_speed_factor"]))
        # 8-connected DIRECTED graph: uphill steeper than max_uphill_slope is blocked, downhill is fine
        idx = np.arange(n * n).reshape(n, n)
        rows, cols, data = [], [], []
        maxs = cfg["max_uphill_slope"]
        self.blocked_edges = 0
        for dy, dx, f in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
            ys = slice(0, n - dy)
            xs = slice(max(0, -dx), n - max(0, dx))
            ys2 = slice(dy, n)
            xs2 = slice(max(0, dx), n - max(0, -dx) if dx < 0 else n)
            a1, a2 = idx[ys, xs], idx[ys2, xs2]
            c = (cost_cell[ys, xs] + cost_cell[ys2, xs2]) / 2 * f
            if self.elev is not None:
                run_cm = self.cell_m * 100 * f
                rise = self.elev[ys2, xs2] - self.elev[ys, xs]          # a1 -> a2
                ok_fwd = rise / run_cm <= maxs
                ok_back = -rise / run_cm <= maxs
            else:
                ok_fwd = ok_back = np.ones_like(c, dtype=bool)
            self.blocked_edges += int((~ok_fwd).sum() + (~ok_back).sum())
            rows += [a1[ok_fwd], a2[ok_back]]
            cols += [a2[ok_fwd], a1[ok_back]]
            data += [c[ok_fwd], c[ok_back]]
        self.graph = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(n * n, n * n))

    def build_elevation(self, xyz, cfg):
        """Ground height per cell from scattered actor Z: low percentile per cell, gaps filled from
        nearest cells, light smoothing. Upper-floor props are ignored by the low percentile."""
        n = self.n
        buckets = {}
        for x, y, z in xyz:
            cell = self.world_to_cell(x, y)
            buckets.setdefault(cell, []).append(z)
        grid = np.full(n * n, np.nan)
        for cell, zs in buckets.items():
            grid[cell] = np.percentile(zs, cfg["elevation_percentile"])
        grid = grid.reshape(n, n)
        known = ~np.isnan(grid)
        print(f"Elevation: {len(xyz)} samples, {known.sum()} of {n * n} cells observed "
              f"({known[self.land].mean() * 100:.0f}% of land cells)")
        if known.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
            grid = grid[iy, ix]
        water_z = np.nanmin(grid) if known.any() else 0.0
        grid = np.where(self.land, grid, np.minimum(grid, water_z))   # water stays at sea level
        return ndimage.gaussian_filter(grid, cfg["elevation_smooth_cells"])

    def world_to_px(self, x, y):
        c = self.cal
        return (c["origin_px"] + x / c["cm_per_px"]) * self.k, (c["origin_py"] + y / c["cm_per_px"]) * self.k

    def on_land(self, x, y):
        px, py = self.world_to_px(x, y)
        ix = int(np.clip(px, 0, self.size - 1)); iy = int(np.clip(py, 0, self.size - 1))
        return bool(self.land_full[iy, ix])

    def world_to_cell(self, x, y):
        px, py = self.world_to_px(x, y)
        cx = int(np.clip(px / self.size * self.n, 0, self.n - 1))
        cy = int(np.clip(py / self.size * self.n, 0, self.n - 1))
        return cy * self.n + cx


def travel_matrix(grid, pts, speed):
    """Seconds between every pair of points: grid shortest path, never less than straight-line time."""
    cells = np.array([grid.world_to_cell(x, y) for x, y in pts])
    uniq, inv = np.unique(cells, return_inverse=True)
    d = dijkstra(grid.graph, directed=True, indices=uniq)         # (len(uniq), n*n), row = from
    T = d[:, uniq][inv][:, inv]
    xy = np.array(pts)
    straight = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) / 100 / speed
    T = np.maximum(T, straight)
    T[~np.isfinite(T)] = 1e6          # unreachable in that direction
    return T


# ----------------------------------------------------------------------------- orienteering heuristic
def route_time(route, T, brk):
    return sum(T[a, b] for a, b in zip(route, route[1:])) + brk * (len(route) - 1)


def _path_time(route, T):
    return sum(T[a, b] for a, b in zip(route, route[1:]))


def two_opt(route, T):
    """Open-path 2-opt for ASYMMETRIC travel times (uphill != downhill): every candidate reversal is
    re-timed in full, because reversing a segment changes the direction of every edge inside it."""
    best = _path_time(route, T)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                cand = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                t = _path_time(cand, T)
                if t < best - 1e-9:
                    route, best, improved = cand, t, True
    return route


def solve(T, prize, budget, brk, fixed_prefix):
    """Greedy best-ratio insertion + 2-opt, repeated until nothing fits. fixed_prefix keeps the
    user's opener (indices into T, starting with 0) in order at the front of the route."""
    route = list(fixed_prefix)
    in_route = set(route)
    locked = len(route)
    candidates = [i for i in range(len(prize)) if i not in in_route and prize[i] > 0]
    while True:
        cur = route_time(route, T, brk)
        best = None
        for c in candidates:
            if c in in_route:
                continue
            # cheapest insertion position (after the locked prefix)
            for pos in range(locked, len(route) + 1):
                a = route[pos - 1]
                b = route[pos] if pos < len(route) else None
                delta = T[a, c] + brk + ((T[c, b] - T[a, b]) if b is not None else 0)
                if cur + delta > budget:
                    continue
                score = prize[c] / max(delta, 1e-3)
                if best is None or score > best[0]:
                    best = (score, c, pos)
        if best is None:
            break
        _, c, pos = best
        route.insert(pos, c)
        in_route.add(c)
        if len(route) - locked > 2:
            tail = two_opt([route[locked - 1]] + route[locked:], T)
            route = route[:locked] + tail[1:]
    return route


# ----------------------------------------------------------------------------- output
def draw(grid, pts, chests, route, out_png, title):
    img = grid.img.copy()
    d = ImageDraw.Draw(img)
    r = max(3, grid.size // 400)
    for c in chests:                                     # all candidate chests, faint
        px, py = grid.world_to_px(c["x"], c["y"])
        col = (200, 200, 200) if not c["car"] else (255, 255, 255)
        d.ellipse([px - r / 2, py - r / 2, px + r / 2, py + r / 2], fill=col)
    path = [grid.world_to_px(*pts[i]) for i in route]
    d.line(path, fill=(255, 40, 40), width=max(2, grid.size // 600))
    for n, (px, py) in enumerate(path):
        rr = r * 1.6
        d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(255, 215, 0) if n else (0, 200, 0), outline=(0, 0, 0))
    d.text((20, 20), title, fill=(255, 255, 255))
    # crop to route area with margin
    xs = [p[0] for p in path]; ys = [p[1] for p in path]
    m = grid.size * 0.06
    box = (max(0, min(xs) - m), max(0, min(ys) - m), min(grid.size, max(xs) + m), min(grid.size, max(ys) + m))
    for im, path in ((img.crop(box), out_png), (img, out_png.replace(".png", "_full.png"))):
        try:
            im.save(path)
        except OSError as ex:   # Windows: file open in a viewer / locked
            alt = path.replace(".png", f"_{int(__import__('time').time())}.png")
            print(f"WARNING: could not write {path} ({ex}); is it open in a viewer? Saved {alt} instead.")
            im.save(alt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loot_csv")
    ap.add_argument("minimap")
    ap.add_argument("--minutes", type=float, default=3.0, help="time budget for the route")
    ap.add_argument("--start", nargs=2, type=float, metavar=("X", "Y"), help="world start (cm)")
    ap.add_argument("--seed-opening", action="store_true", help="force the configured opener first")
    ap.add_argument("--treasure-hour", action="store_true")
    ap.add_argument("--elevation-csv", help="CSV with x,y,z of many actors (e.g. data/discover_all.csv) "
                                            "used to build a terrain height grid; omit = flat terrain")
    ap.add_argument("-o", "--out", default=os.path.join("data", "route"))
    args = ap.parse_args()

    cfg = load_config()
    elev = None
    if args.elevation_csv:
        elev = []
        with open(args.elevation_csv, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                    elev.append((x, y, z))
    grid = Grid(args.minimap, cfg, elev)
    if elev is not None:
        print(f"Slope limit {cfg['max_uphill_slope']}: {grid.blocked_edges} directed grid edges blocked")
        e = grid.elev
        land_e = e[grid.land]
        lo, hi = np.percentile(land_e, 1), np.percentile(land_e, 99)
        g8 = (np.clip((e - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        ep = os.path.join(os.path.dirname(args.out) or ".", "elevation.png")
        Image.fromarray(g8).resize((1024, 1024), Image.NEAREST).save(ep)
        print(f"Elevation preview -> {ep} (bright = high; cliffs should look like sharp edges)")
    chests = load_chests(args.loot_csv, cfg, grid)
    if args.seed_opening:   # user-confirmed reachable: never filter these out
        for sx, sy in cfg["opening"]["stops"]:
            c = min(chests, key=lambda c: math.hypot(c["x"] - sx, c["y"] - sy))
            c["car"] = True
    usable = [c for c in chests if c["car"]]
    print(f"{len(chests)} chest-type locations after dedupe; {len(usable)} car-reachable "
          f"({sum(1 for c in chests if not c['ground_floor'])} above ground floor excluded, "
          f"{sum(1 for c in chests if c['on_water'])} on boats/water excluded, "
          f"{sum(1 for c in chests if not cfg['classes'][c['class']]['car'])} non-car classes excluded)")

    start = args.start or cfg["opening"]["start"]
    pts = [tuple(start)] + [(c["x"], c["y"]) for c in usable]
    prize = [0.0] + [c["prize"] for c in usable]

    print(f"Grid {grid.n}x{grid.n}, cell ~{grid.cell_m:.1f} m; computing travel times for {len(pts)} points...")
    T = travel_matrix(grid, pts, cfg["vehicle_speed_mps"])

    prefix = [0]
    if args.seed_opening:
        xy = np.array(pts)
        for sx, sy in cfg["opening"]["stops"]:
            j = int(np.argmin(np.hypot(xy[:, 0] - sx, xy[:, 1] - sy)))
            if j not in prefix:
                prefix.append(j)
        print(f"Forcing opener of {len(prefix) - 1} stops, "
              f"modelled time {route_time(prefix, T, cfg['break_seconds']):.1f} s (you measured ~15 s)")

    budget = args.minutes * 60
    route = solve(T, prize, budget, cfg["break_seconds"], prefix)
    t = route_time(route, T, cfg["break_seconds"])
    exp_chests = sum(prize[i] for i in route)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "class", "x", "y", "z", "spawn_prob", "arrive_s", "px", "py"])
        clock = 0.0
        for n, i in enumerate(route):
            if n:
                clock += T[route[n - 1], i] + cfg["break_seconds"]
            c = usable[i - 1] if i else {"class": "START", "x": pts[0][0], "y": pts[0][1], "z": 0, "prize": 0}
            px, py = grid.world_to_px(c["x"], c["y"])
            w.writerow([n, c["class"], round(c["x"]), round(c["y"]), round(c["z"]), c["prize"],
                        round(clock, 1), round(px), round(py)])

    print(f"\nRoute: {len(route) - 1} chests in {t:.0f} s of {budget:.0f} s budget; "
          f"expected chests that actually spawn: {exp_chests:.1f}")
    print("Chance per match of each treasure on this route:")
    for name, p in cfg["treasure_odds_per_chest"].items():
        if args.treasure_hour:
            p *= cfg["treasure_hour_multiplier"].get(name, 1.0)
        chance = 1 - (1 - p) ** exp_chests
        print(f"  {name:30s} {chance * 100:8.4f}%   (~1 in {1 / chance:,.0f} matches)")
    title = f"{len(route) - 1} chests / {t:.0f}s / {exp_chests:.1f} expected"
    draw(grid, pts, chests, route, args.out + ".png", title)
    print(f"-> {args.out}.csv, {args.out}.png (cropped), {args.out}_full.png")


if __name__ == "__main__":
    main()
