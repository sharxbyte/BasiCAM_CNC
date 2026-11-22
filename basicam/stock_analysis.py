from dataclasses import dataclass

from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer


@dataclass
class RoughingAnalysis:
    initial_stock_volume_mm3: float
    target_volume_mm3: float
    rough_remaining_shape: object
    rough_removal_shape: object
    finish_remainder_shape: object
    rough_removal_volume_mm3: float
    finish_remainder_volume_mm3: float
    finish_allowance_mm: float


def analyze_roughing(
    stock_shape: object,
    target_shape: object,
    finish_allowance_mm: float = 0.5,
) -> RoughingAnalysis:
    if finish_allowance_mm < 0:
        raise ValueError("Finish allowance cannot be negative.")
    if not _contains_solid(stock_shape):
        raise ValueError("Stock must be a closed solid or compound of solids.")
    if not _contains_solid(target_shape):
        raise ValueError("Target part must be a closed solid or compound of solids.")

    stock_volume = _volume(stock_shape)
    target_volume = _volume(target_shape)
    rough_target = target_shape
    if finish_allowance_mm > 0:
        from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape

        offset = BRepOffsetAPI_MakeOffsetShape()
        offset.PerformByJoin(target_shape, finish_allowance_mm, 1e-5)
        if not offset.IsDone():
            raise ValueError(
                "Open CASCADE could not offset the target for a finish allowance. "
                "Reduce the allowance or inspect tight concave features."
            )
        rough_target = offset.Shape()

    rough_remaining = _boolean(BRepAlgoAPI_Common(stock_shape, rough_target))
    planned_removal = _boolean(BRepAlgoAPI_Cut(stock_shape, rough_remaining))
    finish_remainder = _boolean(BRepAlgoAPI_Cut(rough_remaining, target_shape))
    remaining_volume = _volume(rough_remaining)
    finish_volume = _volume(finish_remainder)
    return RoughingAnalysis(
        initial_stock_volume_mm3=stock_volume,
        target_volume_mm3=target_volume,
        rough_remaining_shape=rough_remaining,
        rough_removal_shape=planned_removal,
        finish_remainder_shape=finish_remainder,
        rough_removal_volume_mm3=max(stock_volume - remaining_volume, 0.0),
        finish_remainder_volume_mm3=finish_volume,
        finish_allowance_mm=finish_allowance_mm,
    )


def _contains_solid(shape: object) -> bool:
    return TopExp_Explorer(shape, TopAbs_SOLID).More()


def _boolean(operation) -> object:
    operation.SetNonDestructive(True)
    operation.Build()
    if not operation.IsDone():
        raise ValueError("Open CASCADE could not complete a stock/target boolean operation.")
    result = operation.Shape()
    if result.IsNull():
        raise ValueError("Stock and target do not overlap as expected for this roughing analysis.")
    if not BRepCheck_Analyzer(result).IsValid():
        raise ValueError("Open CASCADE produced invalid stock remainder geometry.")
    return result


def _volume(shape: object) -> float:
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties, True)
    return float(properties.Mass())