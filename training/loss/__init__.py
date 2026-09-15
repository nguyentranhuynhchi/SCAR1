"""Loss functions for segmentation training."""
from training.loss.losses import DiceLoss, SegmentationLoss
from training.loss.m2_pro_loss import M2ProLoss, present_dice


def build_loss(loss_name="ce_dice", **kwargs):
    """Build the M0-M3 compound loss, returning loss and logging components."""
    normalized = loss_name.lower().replace("-", "_")
    if normalized in {"m2_pro", "m2_pro_loss", "m2pro"}:
        return M2ProLoss(**kwargs)
    if normalized not in {"ce_dice", "segmentation", "segmentation_loss"}:
        raise ValueError(f"Unknown loss {loss_name!r}; expected 'ce_dice' or 'm2_pro_loss'.")
    if "num_classes" in kwargs:
        classes = kwargs.pop("num_classes")
        if "n_classes" in kwargs and kwargs["n_classes"] != classes:
            raise ValueError("Conflicting num_classes and n_classes")
        kwargs["n_classes"] = classes
    return SegmentationLoss(**kwargs)


__all__ = [
    "DiceLoss",
    "SegmentationLoss",
    "M2ProLoss",
    "present_dice",
    "build_loss",
]
