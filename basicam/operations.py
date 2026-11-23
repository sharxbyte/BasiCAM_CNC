from dataclasses import dataclass, field


@dataclass
class Operation:
    name: str
    kind: str
    tool: str
    rpm: int
    feed_mm_min: float
    stepdown_mm: float
    depth_mm: float
    direction: str
    tool_diameter_mm: float = 6.0
    target: str = "Whole part"
    # depth and entry (Block 9)
    first_pass_count: int = 0  # light passes that even the face before roughing
    first_pass_depth_mm: float = 0.5
    final_pass_depth_mm: float = 0.0  # last, finishing depth step (0 = none)
    ramp_mode: str = "Plunge"  # "Plunge", "Angle" or "Length"
    ramp_angle_deg: float = 3.0
    ramp_length_mm: float = 10.0
    lead_in_mm: float = 0.0
    lead_out_mm: float = 0.0
    # contour targets (v2 Block 8): Pocket / Profile / Trace / Drill on sketch contours
    profile_side: str = "Outside"  # Profile: "Inside", "Outside" or "On"
    stepover_ratio: float = 0.6  # Pocket: ring spacing as a fraction of the tool diameter
    peck_mm: float = 0.0  # Drill: peck depth (0 = one plunge)
    aux: list = field(default_factory=list)  # auxiliaries on during this operation (v2 Block 12)
    process_params: dict = field(default_factory=dict)  # laser / plasma / waterjet / EDM cut settings (Block 13)
    wrap: bool = False  # wrap the sketch onto the round stock with the rotary axis (Block 17)
    lathe: dict = field(default_factory=dict)  # turning: allowance, step, groove z/width, thread pitch... (Block 21)


# Fields added in roadmap v2; saved with each operation, old files keep the defaults.
EXTRA_FIELDS = ("profile_side", "stepover_ratio", "peck_mm", "aux", "process_params", "wrap", "lathe")

ENTRY_FIELDS = (
    "first_pass_count", "first_pass_depth_mm", "final_pass_depth_mm", "ramp_mode",
    "ramp_angle_deg", "ramp_length_mm", "lead_in_mm", "lead_out_mm",
)

HARD_MATERIALS = {"steel", "stainless", "stainless steel", "titanium", "tool steel", "cast iron", "inconel"}


def default_entry_params(material: str, finish_quality: str, tool_diameter_mm: float) -> dict:
    """Sensible depth/entry defaults from material and finish quality."""
    hard = material.strip().casefold() in HARD_MATERIALS
    d = max(float(tool_diameter_mm), 0.1)
    finish = finish_quality.strip().casefold()
    final = {"draft": 0.0, "standard": 0.2, "fine": 0.1}.get(finish, 0.2)
    return {
        "first_pass_count": 1 if finish != "draft" else 0,
        "first_pass_depth_mm": round(min(0.5, d * 0.05) if not hard else min(0.25, d * 0.03), 3),
        "final_pass_depth_mm": final * (0.5 if hard else 1.0),
        "ramp_mode": "Angle",
        "ramp_angle_deg": 2.0 if hard else 3.0,
        "ramp_length_mm": round(d * 2.0, 3),
        "lead_in_mm": round(d * 0.5, 3),
        "lead_out_mm": round(d * 0.5, 3),
    }


def pass_depths(total_mm: float, stepdown_mm: float, first_count: int = 0, first_depth_mm: float = 0.5, final_mm: float = 0.0) -> list[float]:
    """Cumulative cut depths: light first passes, roughing at the step-down, then the final pass."""
    total = abs(float(total_mm))
    step = max(0.01, abs(float(stepdown_mm)))
    depths: list[float] = []
    current = 0.0
    for _ in range(max(0, int(first_count))):
        if current >= total - 1e-9:
            break
        current = min(total, current + max(0.01, float(first_depth_mm)))
        depths.append(round(current, 6))
    rough_limit = max(current, total - max(0.0, float(final_mm)))
    while current < rough_limit - 1e-9:
        current = min(rough_limit, current + step)
        depths.append(round(current, 6))
    if current < total - 1e-9:
        depths.append(round(total, 6))
    return depths or [round(total, 6)]


def operation_warnings(operations: list[Operation]) -> list[str]:
    warnings: list[str] = []
    seen_roughing = False
    seen_drilling = False

    for operation in operations:
        kind = operation.kind.casefold()
        if kind in {"finish", "detail", "tap", "ream"} and not seen_roughing:
            warnings.append(
                f"{operation.name}: consider adding a roughing operation before this pass."
            )
        if kind in {"tap", "ream"} and not seen_drilling:
            warnings.append(
                f"{operation.name}: a drilling operation is usually required first."
            )
        seen_roughing |= kind in {"rough", "roughing"}
        seen_drilling |= kind in {"drill", "drilling"}

    return warnings
