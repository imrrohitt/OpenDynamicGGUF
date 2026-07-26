"""Pipeline step IDs — stable names for filesystem checkpoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepDef:
    id: str
    number: int
    title: str
    dir_name: str


# Keep in sync with docs/steps/
STEPS: list[StepDef] = [
    StepDef("resolve", 1, "Resolve model", "01_resolve"),
    StepDef("load", 2, "Load model", "02_load"),
    StepDef("enumerate", 3, "Enumerate tensors", "03_enumerate"),
    StepDef("classify", 4, "Classify tensors", "04_classify"),
    StepDef("catalog", 5, "Build tensor catalog", "05_catalog"),
    StepDef("weight_features", 6, "Weight features", "06_weight_features"),
    StepDef("corpus", 7, "Calibration corpus", "07_corpus"),
    StepDef("activation_features", 8, "Activation features", "08_activation_features"),
    StepDef("freeze_gguf", 9, "Freeze BF16 GGUF", "09_freeze_gguf"),
    StepDef("imatrix", 10, "Build imatrix", "10_imatrix"),
    StepDef("reference_logits", 11, "Cache reference logits", "11_reference_logits"),
    StepDef("sensitivity", 12, "Sensitivity probe", "12_sensitivity"),
    StepDef("optimize", 13, "Optimize recipe", "13_optimize"),
    StepDef("export", 14, "Export GGUF", "14_export"),
    StepDef("validate", 15, "Validate & release", "15_validate"),
]

STEPS_BY_ID = {s.id: s for s in STEPS}


def step_dir_name(step_id: str) -> str:
    return STEPS_BY_ID[step_id].dir_name
