#!/usr/bin/env python3
"""
extract_chests.py - Pull chest placements out of FModel "Save Properties (.json)" exports
of Fortnite World Partition cells and write one clean CSV.

Usage (Windows PowerShell):
    python extract_chests.py "S:\\Tools\\FModel\\Output\\Exports\\FortniteGame\\Plugins\\GameFeatures" -o chests.csv
    python extract_chests.py <root> -o chests.csv --plot chests.png      # also draw a sanity-check scatter

How a chest's position is resolved:
    actor.Properties.RootComponent -> component export (same package, found via the ".<index>" suffix
    of ObjectPath) -> RelativeLocation / RelativeRotation / RelativeScale3D, composed up the
    AttachParent chain until a component with no parent (world space).

Cross-check: Tiered_Chest actors also carry TimeOfDayControlledLightDataArray[].Position, which is the
chest light's baked WORLD position. It sits a few units from the chest root, so a large gap between
the two means the transform resolution went wrong for that chest. The gap is reported per row.

Only the standard library is required (matplotlib only if --plot is used).
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter

CHEST_PREFIX = "Tiered_Chest"          # matches Tiered_Chest_Athena_C, Tiered_Chest_5_Child_C, ...
MAP_PATH_MARKER = "Delilah_Terrain"    # only read cells belonging to the current BR map
MAX_ATTACH_DEPTH = 32                  # guard against cycles in AttachParent chains

# ----------------------------------------------------------------------------- math (UE conventions)

def rotator_to_quat(pitch, yaw, roll):
    """FRotator -> FQuat, same formula as UE's FRotator::Quaternion(). Returns (x, y, z, w)."""
    d = math.pi / 360.0  # deg -> rad, halved
    sp, cp = math.sin(pitch * d), math.cos(pitch * d)
    sy, cy = math.sin(yaw * d), math.cos(yaw * d)
    sr, cr = math.sin(roll * d), math.cos(roll * d)
    return (
        cr * sp * sy - sr * cp * cy,
        -cr * sp * cy - sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_mul(a, b):
    """Hamilton product a*b (applies b first, then a)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    # v' = v + w*t + cross(q.xyz, t)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def quat_yaw_deg(q):
    x, y, z, w = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))  # (loc, quat, scale)


def compose(parent, child):
    """World transform of `child` (given relative to `parent`). Same as UE child * parent
    for the uniform/positive-scale case, which is what placed props use."""
    p_loc, p_rot, p_scl = parent
    c_loc, c_rot, c_scl = child
    scaled = (c_loc[0] * p_scl[0], c_loc[1] * p_scl[1], c_loc[2] * p_scl[2])
    r = quat_rotate(p_rot, scaled)
    loc = (p_loc[0] + r[0], p_loc[1] + r[1], p_loc[2] + r[2])
    return loc, quat_mul(p_rot, c_rot), (p_scl[0] * c_scl[0], p_scl[1] * c_scl[1], p_scl[2] * c_scl[2])

# ----------------------------------------------------------------------------- package helpers

INDEX_RE = re.compile(r"\.(\d+)$")


def ref_index(ref):
    """{'ObjectPath': '/Delilah/.../0C63ZS3K....272'} -> 272 (export index in the same package)."""
    if not isinstance(ref, dict):
        return None
    m = INDEX_RE.search(ref.get("ObjectPath", ""))
    return int(m.group(1)) if m else None


def ref_tail_name(ref):
    """ObjectName "Class'Level:PersistentLevel.Actor.StaticMeshComponent0'" -> 'StaticMeshComponent0'."""
    name = ref.get("ObjectName", "") if isinstance(ref, dict) else ""
    name = name.rstrip("'")
    return name.rsplit(".", 1)[-1] if "." in name else name.split("'")[-1]


def ref_same_package(ref, package_path):
    path = ref.get("ObjectPath", "") if isinstance(ref, dict) else ""
    return path.rsplit(".", 1)[0] == package_path


def resolve_ref(exports, ref, by_name):
    """Find the export a reference points to. Tries the numeric index first (verified by name),
    then falls back to a name lookup (unique names only)."""
    tail = ref_tail_name(ref)
    idx = ref_index(ref)
    if idx is not None:
        for cand in (idx, idx - 1):  # tolerate 0- vs 1-based index differences
            if 0 <= cand < len(exports) and exports[cand].get("Name") == tail:
                return exports[cand]
    hits = by_name.get(tail, [])
    if len(hits) == 1:
        return hits[0]
    # ambiguous name (e.g. many 'StaticMeshComponent0'): match the owning actor in ObjectName
    full = ref.get("ObjectName", "")
    for h in hits:
        outer = h.get("Outer", {})
        outer_name = outer.get("ObjectName", "") if isinstance(outer, dict) else str(outer)
        owner = outer_name.rstrip("'").rsplit(".", 1)[-1]
        if owner and owner in full:
            return h
    return None


def vec(d, keys, default):
    if not isinstance(d, dict):
        return default
    return tuple(float(d.get(k, dv)) for k, dv in zip(keys, default))


def local_transform(comp):
    props = comp.get("Properties", {}) or {}
    # FModel omits properties equal to their default, so a missing value means the default.
    loc = vec(props.get("RelativeLocation"), "XYZ", (0.0, 0.0, 0.0))
    rot = vec(props.get("RelativeRotation"), ("Pitch", "Yaw", "Roll"), (0.0, 0.0, 0.0))
    scl = vec(props.get("RelativeScale3D"), "XYZ", (1.0, 1.0, 1.0))
    has_loc = "RelativeLocation" in props
    return (loc, rotator_to_quat(*rot), scl), has_loc


def world_transform(exports, comp, by_name, package_path):
    """Compose the component's transform up the AttachParent chain. Returns (transform, status)."""
    chain = []
    cur = comp
    status = "ok"
    for _ in range(MAX_ATTACH_DEPTH):
        t, has_loc = local_transform(cur)
        chain.append(t)
        if not has_loc and status == "ok":
            status = "default_location"   # suspicious unless a parent supplies the offset
        parent_ref = (cur.get("Properties") or {}).get("AttachParent")
        if not parent_ref:
            break
        if not ref_same_package(parent_ref, package_path):
            status = "parent_in_other_package"
            break
        parent = resolve_ref(exports, parent_ref, by_name)
        if parent is None:
            status = "parent_unresolved"
            break
        cur = parent
        if status == "default_location":
            status = "ok"  # the offset came from a parent, that's fine
    else:
        status = "attach_chain_too_deep"
    world = IDENTITY
    for t in reversed(chain):          # root-most parent first
        world = compose(world, t)
    return world, status

# ----------------------------------------------------------------------------- extraction

def plugin_of(path):
    parts = re.split(r"[\\/]", path)
    if "GameFeatures" in parts:
        i = parts.index("GameFeatures")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def light_position(actor):
    arr = (actor.get("Properties") or {}).get("TimeOfDayControlledLightDataArray")
    if isinstance(arr, list):
        for e in arr:
            pos = e.get("Position") if isinstance(e, dict) else None
            if isinstance(pos, dict):
                return vec(pos, "XYZ", (0.0, 0.0, 0.0))
    return None


def iter_cell_files(root, generated_only=True):
    for dirpath, _, files in os.walk(root):
        if MAP_PATH_MARKER not in dirpath or (generated_only and "_Generated_" not in dirpath):
            continue
        for f in files:
            if f.lower().endswith(".json"):
                yield os.path.join(dirpath, f)


def package_path_of(exports):
    """Recover '/Delilah/Maps/Delilah_Terrain/_Generated_/XXXX' from any reference in the file."""
    for e in exports:
        outer = e.get("Outer")
        if isinstance(outer, dict) and "ObjectPath" in outer:
            return outer["ObjectPath"].rsplit(".", 1)[0]
    return ""


def extract_file(path, type_re, bbox=None):
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    exports = data if isinstance(data, list) else [data]
    by_name = {}
    for e in exports:
        if isinstance(e, dict):
            by_name.setdefault(e.get("Name"), []).append(e)
    pkg = package_path_of(exports)
    cell = os.path.splitext(os.path.basename(path))[0]
    plugin = plugin_of(path)

    rows = []
    for e in exports:
        if not isinstance(e, dict) or not type_re.search(str(e.get("Type", ""))):
            continue
        props = e.get("Properties") or {}
        if "RootComponent" not in props:   # not a placed actor (components, data objects, ...)
            continue
        light = light_position(e)
        root_ref = props.get("RootComponent")
        comp = resolve_ref(exports, root_ref, by_name) if root_ref else None

        if comp is not None:
            (loc, q, _), status = world_transform(exports, comp, by_name, pkg)
            yaw = quat_yaw_deg(q)
        elif light is not None:
            loc, yaw, status = light, float("nan"), "light_fallback"
        else:
            loc, yaw, status = (float("nan"),) * 3, float("nan"), "unresolved"

        if bbox and not (bbox[0] <= loc[0] <= bbox[2] and bbox[1] <= loc[1] <= bbox[3]):
            continue
        gap = math.dist(loc, light) if (light is not None and comp is not None) else float("nan")
        rows.append({
            "class": e.get("Type"),
            "x": round(loc[0], 2), "y": round(loc[1], 2), "z": round(loc[2], 2),
            "yaw": round(yaw, 2) if not math.isnan(yaw) else "",
            "status": status,
            "light_gap": round(gap, 1) if not math.isnan(gap) else "",
            "plugin": plugin,
            "cell": cell,
            "actor": e.get("Name"),
            "actor_guid": (e.get("ActorInstanceGuid") or {}).get("ActorGuid", ""),
        })
    return rows


FIELDS = ["class", "x", "y", "z", "yaw", "status", "light_gap", "plugin", "cell", "actor", "actor_guid"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder to scan (e.g. ...\\Exports\\FortniteGame\\Plugins\\GameFeatures)")
    ap.add_argument("-o", "--out", default="chests.csv")
    ap.add_argument("--plot", help="optional PNG path for a top-down scatter (needs matplotlib)")
    ap.add_argument("--pattern", default=r"^Tiered_Chest",
                    help="regex on actor class to extract (default: ^Tiered_Chest). "
                         "Example: '^Tiered_Chest|Buried|Floating'")
    ap.add_argument("--discover", action="store_true",
                    help="list EVERY placed actor class (use with --bbox), instead of only --pattern")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                    help="only keep actors inside this world rectangle (cm)")
    ap.add_argument("--all-packages", action="store_true",
                    help="also scan non-_Generated_ packages under the map path (level instances etc.)")
    args = ap.parse_args()
    type_re = re.compile(".*" if args.discover else args.pattern, re.IGNORECASE)

    if os.path.isdir(args.root):
        files = list(iter_cell_files(args.root, generated_only=not args.all_packages))
    else:
        files = [args.root]
    if not files:
        sys.exit(f"No *.json under '{args.root}' inside a {MAP_PATH_MARKER}/_Generated_ folder.")

    rows, bad_files = [], []
    for i, f in enumerate(files, 1):
        try:
            rows.extend(extract_file(f, type_re, args.bbox))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as ex:
            bad_files.append((f, str(ex)))
        if i % 100 == 0:
            print(f"  scanned {i}/{len(files)} files, {len(rows)} matches so far", file=sys.stderr)

    # De-duplicate: the same actor GUID can appear in more than one exported cell.
    # Keep the best-resolved copy of each actor (fully resolved beats light_fallback / unresolved).
    rank = {"ok": 0, "default_location": 1, "parent_in_other_package": 2, "parent_unresolved": 2,
            "attach_chain_too_deep": 2, "light_fallback": 3, "unresolved": 4}
    best = {}
    for r in rows:
        key = r["actor_guid"] or (r["cell"], r["actor"])
        if key not in best or rank.get(r["status"], 9) < rank.get(best[key]["status"], 9):
            best[key] = r
    unique = list(best.values())

    for target in (args.out, args.plot):
        if target and os.path.dirname(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(unique)

    # ---- report
    print(f"\nScanned {len(files)} cell files -> {len(unique)} chests "
          f"({len(rows) - len(unique)} duplicates dropped) -> {args.out}")
    if args.discover:
        print("\nActor classes found (most common first):")
        for cls, n in Counter(r["class"] for r in unique).most_common():
            print(f"  {n:6d}  {cls}")
    else:
        print("By class:   ", dict(Counter(r["class"] for r in unique)))
    print("By plugin:  ", dict(Counter(r["plugin"] for r in unique)))
    print("By status:  ", dict(Counter(r["status"] for r in unique)))
    gaps = sorted(r["light_gap"] for r in unique if r["light_gap"] != "")
    if gaps:
        med = gaps[len(gaps) // 2]
        far = sum(1 for g in gaps if g > 1000)
        print(f"Light cross-check: median gap {med} uu, max {gaps[-1]} uu, {far} chest(s) > 1000 uu "
              f"(large gaps = transform resolution is wrong for those rows)")
    xs = [r["x"] for r in unique if isinstance(r["x"], float) and not math.isnan(r["x"])]
    ys = [r["y"] for r in unique if isinstance(r["y"], float) and not math.isnan(r["y"])]
    if xs:
        print(f"Bounds: X [{min(xs):.0f}, {max(xs):.0f}]  Y [{min(ys):.0f}, {max(ys):.0f}]  (cm)")
    if bad_files:
        print(f"WARNING: {len(bad_files)} file(s) could not be parsed, first: {bad_files[0]}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        palette = ["#2f6fdf", "#e0452b", "#f2a900", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
        for i, (cls, n) in enumerate(Counter(r["class"] for r in unique).most_common()):
            pts = [(r["x"], r["y"]) for r in unique if r["class"] == cls and r["x"] != ""]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=6 if i == 0 else 22,
                       c=palette[i % len(palette)], label=f"{cls} ({n})")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Fortnite top-down: +X = east (right), +Y = south (down)
        ax.set_xlabel("UE X (cm) -> east")
        ax.set_ylabel("UE Y (cm) -> south")
        ax.set_title("Chest placements (top-down)")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Plot -> {args.plot}")


if __name__ == "__main__":
    main()
