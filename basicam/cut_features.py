"""2D cutting features for laser, plasma, waterjet and wire EDM (shapely, no Qt).

Kerf compensation (computed here, no G41/G42); lead-ins and lead-outs whose position and
direction are chosen for the most clearance from other parts and from other features of the
same part, never closer to a neighbouring part's finished outline than half the kerf plus a
safety margin; tabs / micro-joints (default width 2 x kerf); pierce points off the contour;
flash-cut rows for lasers with fast beam gating; cut order (inside first, small parts first,
nearest next + 2-opt); natural-taper compensation; checks that the beam is never on during a
rapid or outside the stock; and the cut segments themselves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

Coord = tuple[float, float]


# ---------------------------------------------------------------- compensation
def compensate(ring: list[Coord], is_hole: bool, kerf: float) -> list[Coord]:
    """Tool-centre path: outer contours move out by kerf/2, holes move in (the part keeps size)."""
    polygon = Polygon(ring)
    shape = polygon.buffer(-kerf / 2.0 if is_hole else kerf / 2.0, join_style=2, mitre_limit=5.0)
    if shape.is_empty:
        return []
    if hasattr(shape, "geoms"):
        shape = max(shape.geoms, key=lambda g: g.area)
    coords = list(shape.exterior.coords)
    return coords


def taper_kerf(kerf: float, thickness: float, taper_deg: float, good_side: str) -> float:
    """Kerf to compensate with for a natural taper: nominal at the bottom, wider at the top by
    2 * t * tan(angle); the chosen good side is kept to size."""
    top = kerf + 2.0 * thickness * math.tan(math.radians(abs(taper_deg)))
    return top if str(good_side).casefold() == "top" else kerf


# ---------------------------------------------------------------- leads
@dataclass
class LeadPlan:
    ring: list[Coord]  # the compensated contour, starting (and ending) at the lead point
    lead_in: list[Coord] = field(default_factory=list)  # from the pierce point to the contour
    lead_out: list[Coord] = field(default_factory=list)
    overcut: list[Coord] = field(default_factory=list)
    ok: bool = True
    message: str = ""
    location: Coord | None = None
    clearance: float = 0.0

    @property
    def pierce(self) -> Coord:
        return self.lead_in[0] if self.lead_in else self.ring[0]


def _corners(ring: list[Coord], sharp_deg: float = 150.0) -> list[Coord]:
    pts = ring[:-1] if ring and ring[0] == ring[-1] else list(ring)
    out = []
    n = len(pts)
    for i in range(n):
        a, b, c = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[(i + 1) % n])
        u, v = a - b, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            continue
        angle = math.degrees(math.acos(min(max(float(np.dot(u, v) / (nu * nv)), -1.0), 1.0)))
        if angle < sharp_deg:
            out.append(tuple(b))
    return out


def _lead_shape(p: np.ndarray, t: np.ndarray, n: np.ndarray, kind: str, length: float, segments: int = 8) -> list[Coord]:
    if kind == "line":
        return [tuple(p + n * length), tuple(p)]
    r = max(length / 2.0, 1e-3)
    c = p + n * r
    arc = [tuple(c - n * r * math.cos(th) - t * r * math.sin(th)) for th in np.linspace(math.pi / 2.0, 0.0, segments + 1)]
    if kind == "arc":
        return arc
    start = np.array(arc[0])
    return [tuple(start + n * (length / 2.0))] + arc  # line + arc


def plan_leads(
    ring: list[Coord],
    is_hole: bool,
    kerf: float,
    lead_type: str = "line+arc",
    lead_in: float = 3.0,
    lead_out: float = 1.0,
    overcut: float = 0.5,
    neighbours: list = (),
    own_features: list = (),
    safety: float = 0.1,
    picked: Coord | None = None,
    samples_per_edge: int = 5,
    within=None,
) -> LeadPlan:
    """Choose the lead position and direction with the most clearance (see the module doc).

    ``ring`` is the compensated contour, ``neighbours`` other parts' finished outlines (shapely
    geometries), ``own_features`` other features of this part (holes, outer edge, tabs...).
    ``picked`` forces the position; it is refused when it breaks the clearance rule."""
    closed = list(ring)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    own = Polygon(closed)
    min_gap = kerf / 2.0 + safety
    obstacles = [g for g in neighbours if g is not None and not g.is_empty]
    features = [g for g in own_features if g is not None and not g.is_empty]
    corners = [Point(c) for c in _corners(closed)]
    kinds = [lead_type] + [k for k in ("line+arc", "arc", "line") if k != lead_type] if lead_type != "none" else ["none"]

    def candidates():
        if picked is not None:
            line = LineString(closed)
            s = line.project(Point(picked))
            yield s
            return
        line = LineString(closed)
        total = line.length
        acc = 0.0
        for a, b in zip(closed[:-1], closed[1:]):
            length = math.dist(a, b)
            for k in range(1, samples_per_edge + 1):
                yield acc + length * k / (samples_per_edge + 1)
            acc += length
        _ = total

    def evaluate(s: float, kind: str, length: float):
        line = LineString(closed)
        p = np.array(line.interpolate(s).coords[0])
        ahead = np.array(line.interpolate(min(s + 1e-3, line.length)).coords[0])
        behind = np.array(line.interpolate(max(s - 1e-3, 0.0)).coords[0])
        t = ahead - behind
        if float(np.linalg.norm(t)) < 1e-12:
            return None
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        inside = own.contains(Point(tuple(p + n * 1e-3)))
        if inside != is_hole:  # point the lead into the scrap: the hole, or outside the part
            n = -n
        if kind == "none":
            pts_in = [tuple(p)]
        else:
            pts_in = _lead_shape(p, t, n, kind, length)
        out_end = p + n * max(lead_out, 0.0)
        pts_out = [tuple(p), tuple(out_end)] if lead_out > 0 else []
        geoms = [LineString(pts_in)] if len(pts_in) >= 2 else [Point(pts_in[0])]
        if pts_out:
            geoms.append(LineString(pts_out))
        lead = unary_union(geoms)
        if is_hole and not own.buffer(1e-6).contains(lead):
            return None
        if not is_hole and own.buffer(-1e-6).intersects(lead):
            return None
        if within is not None and not within.buffer(-kerf / 2.0).contains(lead):
            return None  # the beam may only be on over the stock
        near_neighbour = min((lead.distance(g) for g in obstacles), default=math.inf)
        if near_neighbour < min_gap:
            return ("unsafe", near_neighbour, p)
        near_feature = min((lead.distance(g) for g in features), default=math.inf)
        near_corner = min((Point(tuple(p)).distance(c) for c in corners), default=math.inf)
        score = min(near_neighbour, near_feature, near_corner * 2.0)
        return ("ok", score, p, pts_in, pts_out, s)

    best = None
    unsafe_at = None
    for kind in kinds:
        for factor in (1.0, 0.5, 0.25):
            length = lead_in * factor
            for s in candidates():
                result = evaluate(s, kind, length)
                if result is None:
                    continue
                if result[0] == "unsafe":
                    unsafe_at = unsafe_at or tuple(result[2])
                    continue
                if best is None or result[1] > best[1]:
                    best = result
            if best is not None:
                break
        if best is not None:
            break
    if best is None:
        where = unsafe_at or closed[0]
        why = ("The picked lead position is too close to a neighbouring part" if picked is not None
               else "No safe lead position on this contour")
        return LeadPlan(closed, ok=False, message=f"{why} (needs {min_gap:.3f} mm clearance).", location=where)
    _tag, score, p, pts_in, pts_out, s = best
    line = LineString(closed)
    # restart the ring at the lead point, keep its direction
    total = line.length
    steps = [s] + [line.project(Point(c)) for c in closed[1:-1]]
    ordered = sorted(set(round(v, 9) for v in steps[1:]))
    after = [v for v in ordered if v > s] + [v + total for v in ordered if v <= s]
    ring_pts = [tuple(p)] + [tuple(line.interpolate(v % total).coords[0]) for v in after] + [tuple(p)]
    over = []
    if overcut > 0:
        over = [tuple(p)] + [tuple(line.interpolate((s + overcut * k / 4.0) % total).coords[0]) for k in range(1, 5)]
    return LeadPlan(ring_pts, list(pts_in), list(pts_out), over, True, "", tuple(p), float(score))


# ---------------------------------------------------------------- tabs / micro-joints
def tab_intervals(ring: list[Coord], width: float, count: int = 0, spacing: float = 0.0, keep_clear: list[Coord] = ()) -> list[tuple[float, float]]:
    """Arc-length intervals left uncut, away from corners and from ``keep_clear`` points."""
    line = LineString(ring)
    total = line.length
    if total <= 0 or width <= 0:
        return []
    if count <= 0 and spacing > 0:
        count = max(1, int(total // spacing))
    if count <= 0:
        return []
    avoid = [line.project(Point(c)) for c in list(_corners(ring)) + list(keep_clear)]
    margin = width * 1.5
    out = []
    for k in range(count):
        centre = total * (k + 0.5) / count
        for _ in range(40):
            close = [a for a in avoid if min(abs(centre - a), total - abs(centre - a)) < margin + width / 2.0]
            if not close:
                break
            centre = (centre + width) % total
        out.append((max(0.0, centre - width / 2.0), min(total, centre + width / 2.0)))
    return sorted(out)


def split_by_tabs(ring: list[Coord], intervals: list[tuple[float, float]]) -> list[tuple[str, list[Coord]]]:
    """[("cut" | "tab", points)] along the ring."""
    line = LineString(ring)
    total = line.length
    marks = sorted({0.0, total} | {v for a, b in intervals for v in (a, b)} | {line.project(Point(c)) for c in ring})
    out: list[tuple[str, list[Coord]]] = []
    for a, b in zip(marks[:-1], marks[1:]):
        if b - a < 1e-9:
            continue
        kind = "tab" if any(lo - 1e-9 <= a and b <= hi + 1e-9 for lo, hi in intervals) else "cut"
        pts = [tuple(line.interpolate(a).coords[0]), tuple(line.interpolate(b).coords[0])]
        if out and out[-1][0] == kind:
            out[-1][1].append(pts[1])
        else:
            out.append((kind, pts))
    return out


# ---------------------------------------------------------------- flash cutting
@dataclass
class FlashRow:
    features: list[list[Coord]]
    y: float


def flash_rows(features: list[list[Coord]], feed_mm_min: float, min_gate_s: float, row_tolerance: float = 0.5):
    """Group small closed features into rows (same y), sorted along x. Returns (rows, flagged):
    a feature is flagged when its own cut or the gap to the next one is shorter than the
    distance travelled in the minimum gate time."""
    min_mm = feed_mm_min / 60.0 * max(min_gate_s, 0.0)
    centres = [(Polygon(f).centroid.x, Polygon(f).centroid.y, f) for f in features]
    rows: list[FlashRow] = []
    for x, y, f in sorted(centres, key=lambda c: (c[1], c[0])):
        row = next((r for r in rows if abs(r.y - y) <= row_tolerance), None)
        if row is None:
            rows.append(FlashRow([f], y))
        else:
            row.features.append(f)
    flagged = []
    for row in rows:
        row.features.sort(key=lambda f: Polygon(f).centroid.x)
        for i, f in enumerate(row.features):
            perimeter = LineString(f).length
            gap = math.inf
            if i + 1 < len(row.features):
                gap = Polygon(f).distance(Polygon(row.features[i + 1]))
            if perimeter < min_mm or gap < min_mm:
                flagged.append(f)
    return rows, flagged


# ---------------------------------------------------------------- order
def order_cuts(parts: list[tuple[list[Coord], list[list[Coord]]]], start: Coord = (0.0, 0.0)) -> list[tuple[int, str, list[Coord]]]:
    """Cut sequence [(part index, "inner" | "outer", ring)]: small parts first, nearest next,
    and every part's inner contours before its outer contour (so parts never move before
    their holes are cut)."""
    areas = [Polygon(outer).area for outer, _inner in parts]
    remaining = sorted(range(len(parts)), key=lambda i: areas[i])
    here = np.array(start, dtype=float)
    sequence: list[tuple[int, str, list[Coord]]] = []
    while remaining:
        # small first, then nearest among parts of similar size
        smallest = areas[remaining[0]]
        pool = [i for i in remaining if areas[i] <= smallest * 2.0 + 1e-9]
        index = min(pool, key=lambda i: float(np.linalg.norm(np.array(parts[i][0][0]) - here)))
        remaining.remove(index)
        outer, inners = parts[index]
        todo = list(inners)
        while todo:
            nearest = min(todo, key=lambda r: float(np.linalg.norm(np.array(r[0]) - here)))
            todo.remove(nearest)
            sequence.append((index, "inner", nearest))
            here = np.array(nearest[0])
        sequence.append((index, "outer", outer))
        here = np.array(outer[0])
    return sequence


# ---------------------------------------------------------------- safety checks
def beam_violations(segments, stock_polygon: Polygon | None) -> list[str]:
    """Beam / arc / jet must never be on during a rapid, and only on over the stock."""
    on = False
    out = []
    for index, seg in enumerate(segments, start=1):
        meta = seg.metadata or {}
        if meta.get("beam_on"):
            on = True
        if seg.kind in ("rapid", "retract", "traverse") and on:
            out.append(f"move {index}: the beam is on during a {'gated traverse' if seg.kind == 'traverse' else 'rapid'}")
        if on and seg.kind in ("cut", "lead_in", "lead_out", "pierce") and stock_polygon is not None:
            for p in (seg.start, seg.end):
                if not stock_polygon.buffer(1e-6).contains(Point(p[0], p[1])):
                    out.append(f"move {index}: the beam is on outside the stock at X{p[0]:.3f} Y{p[1]:.3f}")
                    break
        if meta.get("beam_off"):
            on = False
    return out


# ---------------------------------------------------------------- segments
def cut_segments(plans: list[LeadPlan], params, beam_on: list[str], beam_off: list[str], z_top: float = 0.0,
                 safe_z: float = 5.0, tabs: dict | None = None, op_meta: dict | None = None, flash: bool = False) -> list:
    """Segments for planned contours: rapid (beam off) to the pierce point at pierce height,
    pierce (beam on + dwell), down to cut height, lead in, contour (tabs = beam off), overcut,
    lead out, beam off."""
    from basicam.cam_dynamics_post import ToolpathSegment

    meta = dict(op_meta or {})
    feed = max(1.0, float(params.feed_mm_min))
    pierce_z = z_top + float(getattr(params, "pierce_height_mm", 0.0) or 0.0)
    cut_z = z_top + float(getattr(params, "cut_height_mm", 0.0) or 0.0) if params.process == "plasma" else z_top
    dwell = float(params.pierce_delay_s if params.process == "plasma" else params.pierce_time_s if params.process == "waterjet" else 0.0)
    out = []
    here = None

    def add(kind, a, b, f, **extra):
        out.append(ToolpathSegment(kind, (float(a[0]), float(a[1]), float(a[2])), (float(b[0]), float(b[1]), float(b[2])), f, 0, {**meta, **extra}))

    for index, plan in enumerate(plans):
        if not plan.ok or len(plan.ring) < 2:
            continue
        pierce = plan.pierce
        start = here if here is not None else (pierce[0], pierce[1], safe_z)
        if flash and here is not None:
            add("traverse", here, (pierce[0], pierce[1], cut_z), feed, gated=True)  # gated off by the previous move
            add("cut", (pierce[0], pierce[1], cut_z), (pierce[0], pierce[1], cut_z), feed, beam_on=list(beam_on), gated=True)
        else:
            add("rapid", start, (pierce[0], pierce[1], max(safe_z, pierce_z)), 0.0)
            add("rapid", (pierce[0], pierce[1], max(safe_z, pierce_z)), (pierce[0], pierce[1], pierce_z), 0.0)
            add("pierce", (pierce[0], pierce[1], pierce_z), (pierce[0], pierce[1], pierce_z), feed, beam_on=list(beam_on), dwell_s=dwell, contour=index)
            if abs(pierce_z - cut_z) > 1e-9:
                add("plunge", (pierce[0], pierce[1], pierce_z), (pierce[0], pierce[1], cut_z), feed * 0.5, contour=index)
        pts = list(plan.lead_in)
        for a, b in zip(pts[:-1], pts[1:]):
            add("lead_in", (a[0], a[1], cut_z), (b[0], b[1], cut_z), feed * 0.8, contour=index)
        pieces = split_by_tabs(plan.ring, (tabs or {}).get(index, [])) if tabs else [("cut", plan.ring)]
        for kind, coords in pieces:
            for a, b in zip(coords[:-1], coords[1:]):
                if kind == "tab":
                    if out and not out[-1].metadata.get("tab"):
                        out[-1].metadata["beam_off"] = list(beam_off)  # off before the bridge
                    add("traverse", (a[0], a[1], cut_z), (b[0], b[1], cut_z), feed, tab=True, contour=index)
                else:
                    extra = {}
                    if out and out[-1].metadata.get("tab"):
                        extra["beam_on"] = list(beam_on)
                    add("cut", (a[0], a[1], cut_z), (b[0], b[1], cut_z), feed, contour=index, **extra)
        for a, b in zip(plan.overcut[:-1], plan.overcut[1:]):
            add("cut", (a[0], a[1], cut_z), (b[0], b[1], cut_z), feed, overcut=True, contour=index)
        for a, b in zip(plan.lead_out[:-1], plan.lead_out[1:]):
            add("lead_out", (a[0], a[1], cut_z), (b[0], b[1], cut_z), feed * 0.8, contour=index)
        last = plan.lead_out[-1] if plan.lead_out else plan.ring[-1]
        if flash:
            here = (last[0], last[1], cut_z)
            out[-1].metadata["beam_off"] = list(beam_off)
        else:
            out[-1].metadata["beam_off"] = list(beam_off)  # off before the retract, never during it
            add("retract", (last[0], last[1], cut_z), (last[0], last[1], max(safe_z, pierce_z)), 0.0)
            here = (last[0], last[1], max(safe_z, pierce_z))
    if flash and here is not None:
        add("retract", here, (here[0], here[1], safe_z), 0.0)
    return out


__all__ = [
    "FlashRow", "LeadPlan", "beam_violations", "compensate", "cut_segments", "flash_rows", "order_cuts",
    "plan_leads", "split_by_tabs", "tab_intervals", "taper_kerf",
]
