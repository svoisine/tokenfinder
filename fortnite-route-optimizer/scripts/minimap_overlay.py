#!/usr/bin/env python3
"""
minimap_overlay.py - Map chest world coordinates (UE cm) onto the minimap image.

    python scripts/minimap_overlay.py data/chests.csv data/minimap.png -o data/chests_on_map.png
    python scripts/minimap_overlay.py data/chests.csv data/minimap.png -o data/chests_on_map.png --fit

Mapping (Fortnite top-down: world +X = image right/east, world +Y = image DOWN/south):
    px = ORIGIN_PX + X / CM_PER_PX
    py = ORIGIN_PY + Y / CM_PER_PX
for a 2048x2048 minimap. Other image sizes are scaled automatically.

The default constants were estimated by fitting chest positions to the island's land mask.
They are accurate to roughly +/-10 px (about +/-15 m). --fit refines them against your exact CSV and
saves the result to config/minimap_calibration.json, which later runs (and your app) should read.

Also writes <csv>_px.csv: the input rows plus px, py columns for the frontend.
"""
import argparse
import csv
import json
import os

# Estimated for the Chapter 7 (Delilah) 2048x2048 minimap. Re-fit when the map or the image changes.
DEFAULT_CAL = {"cm_per_px": 146.0, "origin_px": 968.0, "origin_py": 1014.0, "ref_size": 2048}
CAL_PATH = os.path.join("config", "minimap_calibration.json")


def load_cal():
    if os.path.exists(CAL_PATH):
        with open(CAL_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return dict(DEFAULT_CAL)


def world_to_px(x, y, cal, img_size):
    k = img_size / cal["ref_size"]
    return (cal["origin_px"] + x / cal["cm_per_px"]) * k, (cal["origin_py"] + y / cal["cm_per_px"]) * k


def read_chests(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                r["_x"], r["_y"] = float(r["x"]), float(r["y"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def land_distance(img):
    """Distance (px) from every pixel to the nearest land pixel, from the minimap's colors."""
    import numpy as np
    from scipy import ndimage
    a = np.asarray(img.convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    land = (r > b + 15) | (g > b + 25)            # green/orange/sand vs. blue/cyan water
    land = ndimage.binary_opening(land, iterations=2)
    land = ndimage.binary_fill_holes(land)
    return ndimage.distance_transform_edt(~land)


def fit(rows, img, cal):
    """Local search around the current calibration that minimizes chest-to-land distance."""
    import numpy as np
    dist = land_distance(img)
    n = img.size[0]
    k = n / cal["ref_size"]
    X = np.array([r["_x"] for r in rows])
    Y = np.array([r["_y"] for r in rows])

    def cost(s, ox, oy):
        px = ((ox + X / s) * k).astype(int)
        py = ((oy + Y / s) * k).astype(int)
        ok = (px >= 0) & (px < n) & (py >= 0) & (py < n)
        d = np.full(len(X), 50.0)
        d[ok] = dist[py[ok], px[ok]]
        return float(np.mean(np.minimum(d, 30)))

    best = (cost(cal["cm_per_px"], cal["origin_px"], cal["origin_py"]),
            cal["cm_per_px"], cal["origin_px"], cal["origin_py"])
    print(f"start cost {best[0]:.3f}")
    # coarse-to-fine; the window stays near the start so the fit can't collapse onto a tiny patch of land
    for step_s, step_o, span in ((2.0, 4.0, 4), (0.5, 1.0, 4), (0.1, 0.25, 4)):
        s0, x0, y0 = best[1:]
        for i in range(-span, span + 1):
            for j in range(-span, span + 1):
                for m in range(-span, span + 1):
                    s, ox, oy = s0 + i * step_s, x0 + j * step_o, y0 + m * step_o
                    c = cost(s, ox, oy)
                    if c < best[0]:
                        best = (c, s, ox, oy)
    print(f"fitted cost {best[0]:.3f}  (mean px from land, capped at 30; lower is better)")
    if abs(best[1] - cal["cm_per_px"]) > 10:
        print("WARNING: scale moved a lot. Check the overlay image before trusting this fit.")
    return {"cm_per_px": round(best[1], 3), "origin_px": round(best[2], 2),
            "origin_py": round(best[3], 2), "ref_size": cal["ref_size"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("minimap")
    ap.add_argument("-o", "--out", default=os.path.join("data", "chests_on_map.png"))
    ap.add_argument("--fit", action="store_true", help="refine calibration and save it to config/")
    args = ap.parse_args()

    from PIL import Image, ImageDraw
    img = Image.open(args.minimap).convert("RGB")
    if img.size[0] != img.size[1]:
        print(f"WARNING: minimap is not square {img.size}; the mapping assumes a square image.")
    rows = read_chests(args.csv)
    cal = load_cal()

    if args.fit:
        cal = fit(rows, img, cal)
        os.makedirs(os.path.dirname(CAL_PATH), exist_ok=True)
        with open(CAL_PATH, "w", encoding="utf-8") as fh:
            json.dump(cal, fh, indent=2)
        print(f"calibration -> {CAL_PATH}: {cal}")

    n = img.size[0]
    draw = ImageDraw.Draw(img)
    rad = max(3, n // 400)
    for r in rows:
        px, py = world_to_px(r["_x"], r["_y"], cal, n)
        r["px"], r["py"] = round(px, 1), round(py, 1)
        special = r.get("class", "") != "Tiered_Chest_Athena_C"
        rr = rad * 2 if special else rad
        draw.ellipse([px - rr, py - rr, px + rr, py + rr],
                     fill=(255, 215, 0) if special else (255, 40, 40), outline=(0, 0, 0))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)

    px_csv = os.path.splitext(args.csv)[0] + "_px.csv"
    fields = [f for f in rows[0].keys() if not f.startswith("_")] if rows else []
    with open(px_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} chests drawn -> {args.out}\npixel coords -> {px_csv}")


if __name__ == "__main__":
    main()
