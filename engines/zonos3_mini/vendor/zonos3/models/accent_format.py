"""Accent adapter files (lingua ``apps.zonos.accent_adapter``, ``zonos-accent-lora/1``, ``/2`` and ``/3``).

An adapter shifts linear weights by ``up @ down``, router balancing biases by a vector and, in ``/2``, every MoE
expert weight by its own ``up_e @ down_e``. A request served with adapter strength alpha runs ``base + alpha *
adapter``. ``load_adapter`` parses one ``<accent>.pt`` (the file stem is the name requests use) into module-keyed
factors; the servers and the reference apply them (zonos3_sgl.models.accent_adapters, zonos3.reference.accents).

Module names are the checkpoint's: ``tools/convert_lingua_checkpoint.py`` keeps every non-expert tensor under its
lingua key. ``TARGETS`` mirrors lingua's ``apps.zonos.lora.DEFAULT_TARGETS``. Exact shifts: ``<router>.balancing_biases``
(float32 quantile thresholds); as in the converter, the legacy ZeroMean parametrization
``.parametrizations.balancing_biases.original`` maps to ``balancing_biases`` with its mean removed, and the training-only
router metrics (``ent_denom``, ``normalized_entropy``) are dropped. Any other exact shift is refused.

Expert factors (``/2``): ``layers.N.feed_forward.experts.{w13,w2}`` carry ``up`` [E, out, r] and ``down`` [E, r, in] in
lingua's SonicExperts layout (w13 rows interleave gate/up), ``shape`` (E, out, in) and ``experts`` E.

Grown vocabulary (``/3``, e.g. a speech-edit LoRA trained on a base grown with a span-end audio id and edit-marker text
rows): ``/2`` plus ``grown`` = ``{"span_end_id", "text_extra_tokens", "rows"}``, the trained values of the rows that base
added (per audio embedder its span-end row [1, dim], for the text embedder its extra rows [extra, dim], for
``multi_output`` each codebook block's span-end row [n_codebooks, dim]). A server grows its model with them once: plain
requests never read or sample the new ids, so they run as before, and the requests that do (the adapter's) see the rows
as trained. ``grow_adapter`` writes a ``/3`` file from lingua's export, folding its exact embedding / head shifts (which
move only the new rows) into them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

FORMAT = "zonos-accent-lora/1"
FORMAT_EXPERTS = "zonos-accent-lora/2"
FORMAT_GROWN = "zonos-accent-lora/3"
FORMATS = (FORMAT, FORMAT_EXPERTS, FORMAT_GROWN)
# SonicExperts weights lingua's apps.zonos.lora adapts per expert.
EXPERT_WEIGHTS = ("w13", "w2")

BALANCING_BIASES = ".balancing_biases"
_LEGACY_BALANCING_BIASES = ".parametrizations.balancing_biases.original"
# Router state the converter drops (training metrics, never used for routing).
_DROPPED_ROUTER_STATE = (".router.ent_denom", ".router.normalized_entropy")

# Mirrors lingua apps.zonos.lora.DEFAULT_TARGETS.
TARGETS = (
    "attention.wq",
    "attention.wkv",
    "attention.wo",
    "attention.gater",
    "attention.in_proj",
    "feed_forward.w_in",
    "feed_forward.w_out",
    "feed_forward.latent_down",
    "feed_forward.latent_up",
    "router.router_linear",
    "multi_output",
)


def lora_target(module: str) -> str | None:
    """The TARGETS entry a module name matches (lingua's suffix rule), or None."""
    for target in TARGETS:
        if module == target or module.endswith("." + target):
            return target
    return None


def served_exact_name(name: str) -> tuple[str | None, bool]:
    """Served tensor an exact shift lands on, and whether to remove its mean.

    Mirrors convert_lingua_checkpoint._output_key / convert_model_state for the
    router state; None means the converter drops the tensor.
    """
    if any(part in name for part in _DROPPED_ROUTER_STATE):
        return None, False
    if name.endswith(_LEGACY_BALANCING_BIASES):
        return name.removesuffix(_LEGACY_BALANCING_BIASES) + BALANCING_BIASES, True
    return name, False


def list_accent_adapters(directory: str | Path) -> list[str]:
    """Adapter names in a directory, in slot order (slot = index + 1)."""
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"accent adapter directory {path} does not exist")
    names = sorted(file.stem for file in path.glob("*.pt"))
    if not names:
        raise FileNotFoundError(f"no accent adapters (*.pt) in {path}")
    return names


@dataclass(frozen=True)
class GrownVocabulary:
    """The vocabulary a ``/3`` adapter needs: the span-end audio id, the extra text rows and the rows they add."""

    span_end_id: int
    text_extra_tokens: int
    rows: dict[str, torch.Tensor]  # tensor name -> float32 new rows (see the module docstring)


@dataclass(frozen=True)
class AccentAdapter:
    """One parsed adapter.

    ``linear``: LoRA module (e.g. ``layers.3.attention.wq``) -> (down [K, in], up [K, out]) float32, the shift being
    ``(x @ down.T) @ up`` on the flat ``[out, in]`` product; ``shapes``: the weight shape each linear shift was trained
    on. ``experts``: experts module (``layers.N.feed_forward.experts``) -> weight (``w13`` / ``w2``) -> (up [E, out, r],
    down [E, r, in]) float32 in the SonicExperts layout. ``router_bias``: router module (``layers.N.feed_forward.router``)
    -> float32 [experts] added to its balancing biases. ``grown`` (``/3`` only): the vocabulary to grow the model to."""

    name: str
    format: str
    linear: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    experts: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    router_bias: dict[str, torch.Tensor] = field(default_factory=dict)
    grown: GrownVocabulary | None = None

    @property
    def rank(self) -> int:
        """Columns the adapter's linear factors take (their largest rank)."""
        return max((down.shape[0] for down, _ in self.linear.values()), default=0)


def load_adapter(path: str | Path) -> AccentAdapter:
    """Parse ``<accent>.pt``; raises ValueError for a file no server can apply."""
    path = Path(path)
    adapter = torch.load(path, map_location="cpu", weights_only=True)
    fmt = adapter.get("format")
    if fmt not in FORMATS:
        raise ValueError(f"{path}: not a {' / '.join(FORMATS)} adapter")
    grown = None
    if fmt == FORMAT_GROWN:
        g = adapter["grown"]
        grown = GrownVocabulary(
            int(g["span_end_id"]), int(g["text_extra_tokens"]), {name: rows.float() for name, rows in g["rows"].items()}
        )
    router_bias: dict[str, torch.Tensor] = {}
    for target, shift in adapter["exact"].items():
        served, zero_mean = served_exact_name(target)
        if served is None:
            continue
        if not served.endswith(".router" + BALANCING_BIASES):
            raise ValueError(
                f"{path}: shifts {target}; accent serving applies only linear factors and router balancing biases"
            )
        shift = shift.float()
        if zero_mean:
            shift = shift - shift.mean()
        router_bias[served.removesuffix(BALANCING_BIASES)] = shift
    linear: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    experts: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for target, pair in adapter["factors"].items():
        if ".experts." in target or "experts" in pair:
            if fmt == FORMAT:
                raise ValueError(f"{path}: {FORMAT} adapter shifts MoE expert weights ({target})")
            module, _, weight = target.rpartition(".")
            if not module.endswith("feed_forward.experts") or weight not in EXPERT_WEIGHTS:
                raise ValueError(
                    f"accent adapter shifts expert tensor {target!r}; servable expert factors are "
                    f"<layer>.feed_forward.experts.{{{','.join(EXPERT_WEIGHTS)}}}"
                )
            up, down = pair["up"], pair["down"]
            shape = tuple(pair["shape"])
            if (
                len(shape) != 3
                or pair.get("experts") != shape[0]
                or tuple(up.shape) != (shape[0], shape[1], down.shape[-2])
                or tuple(down.shape) != (shape[0], down.shape[-2], shape[2])
            ):
                raise ValueError(
                    f"{path.stem}: {target} factors up {tuple(up.shape)}, down {tuple(down.shape)}, shape {shape}, "
                    f"experts {pair.get('experts')} do not shift the [E, out, in] weight {shape}"
                )
            experts.setdefault(module, {})[weight] = (up.float(), down.float())
            continue
        if not target.endswith(".weight"):
            raise ValueError(f"accent adapter factor {target!r} is not a weight")
        module = target.removesuffix(".weight")
        if lora_target(module) is None:
            raise ValueError(f"accent adapter shifts {module!r}, outside the LoRA targets {', '.join(TARGETS)}")
        linear[module] = (pair["down"].float(), pair["up"].float().T)
        shapes[module] = tuple(pair["shape"])
    return AccentAdapter(path.stem, fmt, linear, shapes, experts, router_bias, grown)


def fit_head(adapter: AccentAdapter, head_rows: int) -> AccentAdapter:
    """``adapter`` on an output head of ``head_rows`` rows. An adapter trained before the vocabulary grew (one span-end
    row per codebook block, a ``/3`` adapter's) shifts the head's old rows only: its ``multi_output`` factor gains zero
    up columns at each block's new row. Any other mismatch is left for the caller's shape check."""
    shape = adapter.shapes.get("multi_output")
    if shape is None or shape[0] >= head_rows or shape[0] % (head_rows - shape[0]):
        return adapter
    n_codebooks = head_rows - shape[0]
    down, up = adapter.linear["multi_output"]
    k = up.shape[0]
    up = torch.cat((up.view(k, n_codebooks, -1), up.new_zeros(k, n_codebooks, 1)), 2).reshape(k, head_rows)
    return replace(
        adapter,
        linear={**adapter.linear, "multi_output": (down, up)},
        shapes={**adapter.shapes, "multi_output": (head_rows, *shape[1:])},
    )


def adapter_format(path: str | Path) -> str | None:
    """An adapter file's ``format`` (memory-mapped read, no tensors loaded)."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True).get("format")


def edit_adapter(directory: str | Path | None) -> str | None:
    """The speech-edit adapter of a directory: its one ``/3`` adapter (edit fills run on it at strength 1, and it is not
    offered as an accent); None without one."""
    names = [
        n
        for n in (list_accent_adapters(directory) if directory else ())
        if adapter_format(Path(directory) / f"{n}.pt") == FORMAT_GROWN
    ]
    if len(names) > 1:
        raise ValueError(f"{directory}: more than one {FORMAT_GROWN} adapter ({', '.join(names)}); edit takes one")
    return names[0] if names else None


def grown_vocabulary(directory: str | Path | None) -> GrownVocabulary | None:
    """The vocabulary a directory's ``/3`` adapter (``edit_adapter``) grows the model to; None without one."""
    name = edit_adapter(directory)
    if name is None:
        return None
    g = torch.load(Path(directory) / f"{name}.pt", map_location="cpu", weights_only=True, mmap=True)["grown"]
    return GrownVocabulary(
        int(g["span_end_id"]), int(g["text_extra_tokens"]), {key: rows.float() for key, rows in g["rows"].items()}
    )


def grow_state_dict(state: dict[str, torch.Tensor], grown: GrownVocabulary, *, n_codebooks: int) -> dict:
    """A checkpoint state dict with ``grown``'s rows appended (``multi_output.weight``: one per codebook block), in
    each tensor's dtype; the existing rows are untouched."""
    for key, rows in grown.rows.items():
        if key not in state:
            raise ValueError(f"grown vocabulary adds rows to {key}, which the checkpoint lacks")
        weight = state[key]
        rows = rows.to(weight.device, weight.dtype)
        if key == "multi_output.weight":
            blocks = weight.view(n_codebooks, -1, weight.shape[-1])
            state[key] = torch.cat((blocks, rows[:, None]), 1).reshape(-1, weight.shape[-1])
        else:
            state[key] = torch.cat((weight, rows))
    return state


def grow_adapter(adapter: dict, rows_before: dict[str, int], grown: dict) -> dict:
    """A ``/3`` file's content from lingua's ``/2`` export of a LoRA trained on a grown-vocabulary base.

    ``grown`` is ``{"span_end_id", "text_extra_tokens", "rows"}`` with the rows that base added as initialized (laid out
    as in the module docstring); ``rows_before`` gives each grown tensor's row count before growing (per codebook block
    for ``multi_output.weight``). The export's exact shifts of those tensors are folded into the new rows; a shift of
    any row the base already had is refused (it would change plain requests)."""
    if adapter.get("format") != FORMAT_EXPERTS:
        raise ValueError(f"grow_adapter takes a {FORMAT_EXPERTS} adapter, got {adapter.get('format')!r}")
    rows = {name: value.float().clone() for name, value in grown["rows"].items()}
    exact = {}
    for name, shift in adapter["exact"].items():
        if name not in rows:
            exact[name] = shift
            continue
        shift, before = shift.float(), rows_before[name]
        if name == "multi_output.weight":
            blocks = rows[name].shape[0]
            shift = shift.view(blocks, -1, shift.shape[-1])
            old, new = shift[:, :before], shift[:, before:].reshape(-1, shift.shape[-1])
        else:
            old, new = shift[:before], shift[before:]
        if old.any():
            raise ValueError(f"grow_adapter: {name} shifts rows the base had; only its new rows may move")
        if new.shape != rows[name].shape:
            raise ValueError(f"grow_adapter: {name} new-row shift {tuple(new.shape)} != rows {tuple(rows[name].shape)}")
        rows[name] += new
    return {
        **adapter,
        "format": FORMAT_GROWN,
        "exact": exact,
        "grown": {
            "span_end_id": int(grown["span_end_id"]),
            "text_extra_tokens": int(grown["text_extra_tokens"]),
            "rows": {name: value.contiguous() for name, value in rows.items()},
        },
    }
