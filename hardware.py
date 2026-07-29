"""Hardware-aware budgeting — feature 01 of the platform (docs/platform/01-…).

Users describe hardware ("I have a 12 GB GPU"), not byte budgets. This module
translates a hardware profile + target context length into the weight-byte
budget the step-13 optimizer already accepts (``--budget-mb``).

    budget = pool × usable_fraction − kv_cache(model, ctx) − runtime_overhead

Nothing here optimizes anything: it is a front-end to the existing knapsack.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA = "odg/hardware/v1"

# Memory the runtime itself needs on top of weights + KV cache
# (compute graph, scratch buffers, tokenizer, mmap slack).
RUNTIME_OVERHEAD_GB = 0.35

# Default usable fraction of the memory pool, by kind. GPUs keep headroom for
# the driver/display; Apple Silicon shares unified memory with the OS; plain
# CPU boxes share RAM with everything else.
DEFAULT_USABLE_FRACTION = {
    "gpu": 0.90,
    "apple_silicon": 0.70,
    "cpu": 0.60,
}

_KV_DTYPE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625}


@dataclass(frozen=True)
class HardwareProfile:
    """One machine the model must fit on. ``odg/hardware/v1``."""

    id: str
    kind: str  # gpu | apple_silicon | cpu
    vram_gb: float  # 0 for cpu / apple_silicon (unified pool lives in ram_gb)
    ram_gb: float
    bandwidth_gbps: float | None = None
    usable_fraction: float | None = None  # None → kind default
    source: str = "named"  # named | flags | detected
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in DEFAULT_USABLE_FRACTION:
            raise ValueError(
                f"Unknown hardware kind {self.kind!r} "
                f"(expected one of {sorted(DEFAULT_USABLE_FRACTION)})"
            )
        if self.memory_pool_gb() <= 0:
            raise ValueError(f"Profile {self.id!r} has no usable memory pool")

    def memory_pool_gb(self) -> float:
        """The pool weights must fit in: VRAM for GPUs, RAM otherwise."""
        return self.vram_gb if self.kind == "gpu" else self.ram_gb

    def effective_usable_fraction(self) -> float:
        if self.usable_fraction is not None:
            return float(self.usable_fraction)
        return DEFAULT_USABLE_FRACTION[self.kind]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["notes"] = list(self.notes)
        return d


def _gpu(id_: str, vram: float, ram: float, bw: float) -> HardwareProfile:
    return HardwareProfile(id=id_, kind="gpu", vram_gb=vram, ram_gb=ram, bandwidth_gbps=bw)


def _mac(id_: str, ram: float, bw: float) -> HardwareProfile:
    return HardwareProfile(id=id_, kind="apple_silicon", vram_gb=0, ram_gb=ram, bandwidth_gbps=bw)


def _cpu(id_: str, ram: float) -> HardwareProfile:
    return HardwareProfile(id=id_, kind="cpu", vram_gb=0, ram_gb=ram, bandwidth_gbps=None)


# Built-in named device database (--device <id>). Bandwidth is memory
# bandwidth in GB/s — used later by the speed objective, informational today.
DEVICES: dict[str, HardwareProfile] = {
    p.id: p
    for p in (
        _gpu("rtx-3060-12gb", 12, 32, 360),
        _gpu("rtx-3070-8gb", 8, 32, 448),
        _gpu("rtx-3080-10gb", 10, 32, 760),
        _gpu("rtx-3090-24gb", 24, 64, 936),
        _gpu("rtx-4060-8gb", 8, 32, 272),
        _gpu("rtx-4070-12gb", 12, 32, 504),
        _gpu("rtx-4080-16gb", 16, 64, 717),
        _gpu("rtx-4090-24gb", 24, 64, 1008),
        _gpu("rx-7900-xtx-24gb", 24, 64, 960),
        _mac("mac-8gb", 8, 100),
        _mac("mac-16gb", 16, 120),
        _mac("macbook-air-16gb", 16, 120),
        _mac("mac-32gb", 32, 200),
        _mac("mac-64gb", 64, 400),
        _mac("mac-128gb", 128, 800),
        _cpu("cpu-16gb", 16),
        _cpu("cpu-32gb", 32),
        _cpu("cpu-64gb", 64),
    )
}


def parse_size_gb(value: str | float | int) -> float:
    """``"24GB"`` / ``"16GiB"`` / ``"12.5g"`` / ``24`` → gigabytes as float."""
    if isinstance(value, (int, float)):
        gb = float(value)
    else:
        m = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*(gib|gb|g|mib|mb|m)?\s*",
            str(value),
            re.IGNORECASE,
        )
        if not m:
            raise ValueError(f"Cannot parse memory size {value!r} (try e.g. '24GB')")
        num = float(m.group(1))
        unit = (m.group(2) or "gb").lower()
        gb = num / 1024.0 if unit in ("mib", "mb", "m") else num
    if gb <= 0:
        raise ValueError(f"Memory size must be positive, got {value!r}")
    return gb


def get_device(device_id: str) -> HardwareProfile:
    key = device_id.strip().lower()
    if key not in DEVICES:
        known = ", ".join(sorted(DEVICES))
        raise ValueError(f"Unknown device {device_id!r}. Known devices: {known}")
    return DEVICES[key]


def profile_from_flags(
    *,
    gpu: str | None = None,
    ram: str | None = None,
    device: str | None = None,
    cpu_only: bool = False,
) -> HardwareProfile:
    """Build a profile from CLI intent flags. Priority: --device > --gpu/--ram."""
    if device:
        return get_device(device)
    if gpu and not cpu_only:
        vram = parse_size_gb(gpu)
        ram_gb = parse_size_gb(ram) if ram else max(vram * 2, 16.0)
        return HardwareProfile(
            id=f"gpu-{vram:g}gb",
            kind="gpu",
            vram_gb=vram,
            ram_gb=ram_gb,
            source="flags",
        )
    if ram:
        ram_gb = parse_size_gb(ram)
        kind = "cpu" if cpu_only or platform.system() != "Darwin" else "apple_silicon"
        return HardwareProfile(
            id=f"{kind.replace('_', '-')}-{ram_gb:g}gb",
            kind=kind,
            vram_gb=0,
            ram_gb=ram_gb,
            source="flags",
        )
    raise ValueError("Describe your hardware: --device <id>, --gpu <size>, or --ram <size>")


def detect_profile() -> HardwareProfile | None:
    """Best-effort local probe. Returns None when nothing can be detected."""
    # NVIDIA GPU
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            mib = float(out.stdout.strip().splitlines()[0])
            vram = round(mib / 1024.0, 1)
            return HardwareProfile(
                id=f"detected-gpu-{vram:g}gb",
                kind="gpu",
                vram_gb=vram,
                ram_gb=max(vram * 2, 16.0),
                source="detected",
                notes=("VRAM detected via nvidia-smi; system RAM assumed.",),
            )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass

    # Apple Silicon unified memory
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                ram = round(int(out.stdout.strip()) / (1024**3), 1)
                return HardwareProfile(
                    id=f"detected-mac-{ram:g}gb",
                    kind="apple_silicon",
                    vram_gb=0,
                    ram_gb=ram,
                    source="detected",
                    notes=("Unified memory detected via sysctl hw.memsize.",),
                )
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass

    # Plain RAM (Linux)
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    ram = round(kb / (1024**2), 1)
                    return HardwareProfile(
                        id=f"detected-cpu-{ram:g}gb",
                        kind="cpu",
                        vram_gb=0,
                        ram_gb=ram,
                        source="detected",
                        notes=("RAM detected via /proc/meminfo.",),
                    )
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# KV cache + budget derivation
# ---------------------------------------------------------------------------


def kv_cache_bytes(
    descriptor: dict[str, Any],
    ctx: int,
    kv_dtype: str = "f16",
) -> int | None:
    """
    Exact-math KV cache size from the architecture descriptor (step 01 output).

    bytes = 2 (K+V) × layers × ctx × kv_dim × dtype_bytes

    Uses head-count fields when the descriptor has them; otherwise falls back
    to embedding_length as the KV width (exact for MHA, an overestimate for
    GQA models — conservative, so the model still fits).
    """
    layers = descriptor.get("layer_count")
    embd = descriptor.get("embedding_length")
    if not layers or not embd:
        return None

    dtype_bytes = _KV_DTYPE_BYTES.get(kv_dtype.lower())
    if dtype_bytes is None:
        raise ValueError(f"Unknown KV dtype {kv_dtype!r} (expected f16/f32/q8_0)")

    head_count = descriptor.get("head_count") or descriptor.get("attention_head_count")
    head_count_kv = descriptor.get("head_count_kv") or descriptor.get(
        "attention_head_count_kv"
    )
    if head_count and head_count_kv:
        head_dim = int(embd) // int(head_count)
        kv_dim = int(head_count_kv) * head_dim
    else:
        kv_dim = int(embd)

    return int(2 * int(layers) * int(ctx) * kv_dim * dtype_bytes)


@dataclass
class BudgetPlan:
    """The auditable translation from hardware intent to optimizer budget."""

    profile: dict[str, Any]
    ctx: int
    kv_dtype: str
    pool_gb: float
    usable_fraction: float
    usable_gb: float
    kv_cache_gb: float
    kv_cache_exact: bool  # False when descriptor lacked dims and we reserved a default
    runtime_overhead_gb: float
    weight_budget_bytes: int
    weight_budget_mb: float
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = "odg/fitplan/v1"
        return d

    def display_rows(self) -> list[tuple[str, str]]:
        """Every subtraction shown, so the budget is auditable."""
        return [
            ("profile", f"{self.profile['id']} ({self.profile['kind']}, {self.profile['source']})"),
            ("memory pool", f"{self.pool_gb:.1f} GB"),
            ("usable", f"{self.usable_gb:.2f} GB (fraction {self.usable_fraction:.2f})"),
            (
                "kv cache",
                f"−{self.kv_cache_gb:.2f} GB (ctx {self.ctx}, {self.kv_dtype}"
                + ("" if self.kv_cache_exact else ", reserved default")
                + ")",
            ),
            ("runtime overhead", f"−{self.runtime_overhead_gb:.2f} GB"),
            (
                "weight budget",
                f"{self.weight_budget_bytes / (1024**3):.2f} GB "
                f"→ --budget-mb {self.weight_budget_mb:.0f}",
            ),
        ]


def derive_budget(
    profile: HardwareProfile,
    descriptor: dict[str, Any],
    *,
    ctx: int | None = None,
    kv_dtype: str = "f16",
) -> BudgetPlan:
    """profile + architecture descriptor + ctx → weight-byte budget."""
    notes: list[str] = list(profile.notes)

    model_ctx = descriptor.get("context_length")
    if ctx is None:
        ctx = min(4096, int(model_ctx)) if model_ctx else 4096
        notes.append(f"Context defaulted to {ctx} (pass --ctx to change).")
    elif model_ctx and int(ctx) > int(model_ctx):
        notes.append(
            f"Requested ctx {ctx} exceeds the model's trained context "
            f"{model_ctx}; sizing for {ctx} anyway."
        )

    pool_gb = profile.memory_pool_gb()
    usable_fraction = profile.effective_usable_fraction()
    usable_gb = pool_gb * usable_fraction

    kv_bytes = kv_cache_bytes(descriptor, int(ctx), kv_dtype)
    kv_exact = kv_bytes is not None
    if kv_bytes is None:
        kv_bytes = int(0.5 * 1024**3)
        notes.append(
            "Descriptor lacks layer/embedding dims — reserved a default 0.5 GB "
            "for KV cache instead of exact math."
        )
    elif not (
        descriptor.get("head_count_kv") or descriptor.get("attention_head_count_kv")
    ):
        notes.append(
            "KV cache sized with full embedding width (no KV-head count in "
            "descriptor) — an overestimate for GQA models, so this is safe."
        )

    kv_gb = kv_bytes / (1024**3)
    budget_gb = usable_gb - kv_gb - RUNTIME_OVERHEAD_GB
    budget_bytes = int(budget_gb * 1024**3)
    if budget_bytes <= 0:
        raise ValueError(
            f"No room for weights on {profile.id}: pool {pool_gb:.1f} GB × "
            f"{usable_fraction:.2f} usable − {kv_gb:.2f} GB KV (ctx {ctx}) − "
            f"{RUNTIME_OVERHEAD_GB:.2f} GB runtime ≤ 0. "
            "Lower --ctx or pick smaller hardware targets."
        )

    return BudgetPlan(
        profile=profile.to_dict(),
        ctx=int(ctx),
        kv_dtype=kv_dtype,
        pool_gb=pool_gb,
        usable_fraction=usable_fraction,
        usable_gb=usable_gb,
        kv_cache_gb=kv_gb,
        kv_cache_exact=kv_exact,
        runtime_overhead_gb=RUNTIME_OVERHEAD_GB,
        weight_budget_bytes=budget_bytes,
        weight_budget_mb=budget_bytes / (1024**2),
        notes=notes,
    )


def list_devices_rows() -> list[dict[str, str]]:
    rows = []
    for p in sorted(DEVICES.values(), key=lambda x: (x.kind, x.memory_pool_gb())):
        rows.append(
            {
                "id": p.id,
                "kind": p.kind,
                "pool": f"{p.memory_pool_gb():g} GB",
                "bandwidth": f"{p.bandwidth_gbps:g} GB/s" if p.bandwidth_gbps else "-",
                "usable": f"{p.effective_usable_fraction():.0%}",
            }
        )
    return rows
