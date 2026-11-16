"""Steering bundles: several direction sets served together, driven by one request field.

A bundle file (``--steer-bundle``) names direction sets and carries the UI metadata
the studio reads from ``/v2/controls``::

    {"name": "studio_mect60k_x10",
     "sets": {"identity": "speaker_attributes_mect60k_x10", "delivery": "ears_styles_mect60k_x10", ...},
     "timing": {"delivery.tempo": [[-2, 2.2], [0, 1.0], [2, 0.63]]},
     "ui": {...}}

Set directories are relative to the bundle file. A request's ``controls`` maps
``<set>.<direction>`` to a raw dose. Directions of ``raw`` sets move the speaker
embedding (their deltas summed, the result rescaled to the embedding's norm);
``attnres_source`` directions add to their Full
AttnRes bank source on every generated frame. ``control_spans`` apply hidden
directions to the frames generated while a character range of the text is spoken:
frames are placed by character position at ``frames_per_char`` (the client's
measurement from an earlier take, or a default), stretched inside each span by the
``timing`` table of the controls that change the pace, with linear ramps of
``RAMP_FRAMES`` at both edges. ``control_span_follow: "reading"`` places the same
span deltas by the model's reading position instead (zonos3.tts.span_follow).

``steer_release`` (bundle key, ``ReleaseSchedule``; request fields ``steer_hold`` /
``steer_onset_seconds`` / ``steer_ramp_seconds``) weights the request-level hidden
``controls`` over the generated frames: full dose for ``onset`` frames, a linear ramp to
``hold`` over ``ramp`` frames, ``hold`` after (``release_weight``). The model copies the
style of the speech it has generated, so a constant push integrates over a long line
(pitch and loudness climb); a held fraction keeps the style without the climb. The
bundle's ``exempt`` directions (pace: span placement and edit lengths assume it) stay
constant, as do control_spans and raw directions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Dict, List, Mapping, Sequence

import numpy as np
import torch

from .directions import HIDDEN_SPACE, RAW_SPACE, DirectionSet

RAMP_FRAMES = 3  # 0.2 s at 15 frames per second
DEFAULT_FRAMES_PER_CHAR = 1.0  # release LM, embedding mode, unsteered (15 fps, ~15 characters per second)


@dataclass
class Direction:
    id: str
    set: str
    name: str
    space: str
    vector: torch.Tensor
    source_index: int | None


@dataclass(frozen=True)
class RateExtension:
    """Tempo-steered speaking-rate buckets past the condition row (bundle ``rate_extension``;
    docs/guide.md "Extended speaking-rate buckets"): the pace direction, the row edges they continue, the nominal edges past
    the row (``below[k]`` the lower edge of bucket -k, ``above[k]`` the upper edge of bucket n-1+k for the row's n
    buckets) and per TTS language the buckets measured safe with their dose (bucket -> dose, contiguous from the row).
    ``reference_sides`` ("slower" / "faster"): the sides that also serve reference-mode voices (a cloned source voice);
    the buckets were validated in embedding mode."""

    direction: str
    row_edges: tuple[float, ...]
    below: tuple[float, ...]
    above: tuple[float, ...]
    doses: Dict[str, Dict[int, float]]
    reference_sides: tuple[str, ...] = ("slower", "faster")


@dataclass(frozen=True)
class ReleaseSchedule:
    """Steer then hold (bundle ``steer_release``): the request-level hidden controls at full dose for ``onset_seconds``
    of generated speech, a linear ramp to ``hold`` x dose over ``ramp_seconds``, ``hold`` after. ``exempt``: hidden
    directions that stay constant. The bundle's values are the server defaults of the request fields; hold 1 is the
    constant push of before."""

    onset_seconds: float
    ramp_seconds: float
    hold: float
    exempt: tuple[str, ...] = ()

    def frames(self, frame_rate: float) -> tuple[int, int, float]:
        """(onset frames, ramp frames, hold) at ``frame_rate`` generated frames per second."""
        return round(self.onset_seconds * frame_rate), round(self.ramp_seconds * frame_rate), float(self.hold)


def check_release(onset_seconds: float, ramp_seconds: float, hold: float) -> None:
    """ValueError unless 0 <= hold <= 1 and the times are finite and >= 0."""
    if not (np.isfinite(hold) and 0.0 <= hold <= 1.0):
        raise ValueError(f"steer_hold must be in [0, 1], got {hold!r}.")
    for name, v in (("steer_onset_seconds", onset_seconds), ("steer_ramp_seconds", ramp_seconds)):
        if not (np.isfinite(v) and v >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0, got {v!r}.")


@dataclass
class SteeringBundle:
    name: str
    path: Path
    sets: Dict[str, DirectionSet]
    directions: Dict[str, Direction]
    timing: Dict[str, List[List[float]]] = field(default_factory=dict)
    ui: dict = field(default_factory=dict)
    rate_extension: RateExtension | None = None
    steer_release: ReleaseSchedule | None = None

    @property
    def hidden_sources(self) -> List[int]:
        return sorted({int(d.source_index) for d in self.sets.values() if d.space == HIDDEN_SPACE})

    @property
    def raw_dim(self) -> int | None:
        dims = {int(d.dim) for d in self.sets.values() if d.space == RAW_SPACE}
        return dims.pop() if dims else None


def no_steering_message(hint: str, *, field: str | None = None) -> str:
    """The error of a steering request on a server without a bundle (``hint``: how to load one; ``field``: the request
    field that needs it)."""
    if field is None:
        return f"This server has no steering bundle ({hint})."
    return f"This server has no steering bundle ({hint}); `{field}` are unavailable."


def build_bundle(
    origin: Path,
    name: str,
    sets: Dict[str, DirectionSet],
    *,
    timing: dict,
    ui: dict,
    rate_extension: dict | None,
    steer_release: dict | None = None,
) -> SteeringBundle:
    """A bundle of loaded direction sets (``origin``: the bundle file, for messages and ``path``): every check of the
    bundle file format (set names, non-empty sets, one raw embedding dim, timing of known directions, a hidden
    rate_extension direction, steer_release values and hidden exempt directions), the directions, the rate extension
    and the steer-then-hold defaults."""
    directions: Dict[str, Direction] = {}
    raw_dims = set()
    for set_name, loaded in sets.items():
        if "." in set_name:
            raise ValueError(f"{origin}: set name {set_name!r} must not contain '.'")
        if loaded.is_empty():
            raise ValueError(f"{origin}: set {set_name!r} has no directions")
        if loaded.space == RAW_SPACE:
            raw_dims.add(int(loaded.dim))
        for direction, vec in {**loaded.named, **loaded.axes}.items():
            did = f"{set_name}.{direction}"
            directions[did] = Direction(
                did,
                set_name,
                direction,
                loaded.space,
                vec,
                loaded.source_index if loaded.space == HIDDEN_SPACE else None,
            )
    if len(raw_dims) > 1:
        raise ValueError(f"{origin}: raw sets disagree on the embedding dim: {sorted(raw_dims)}")
    unknown = sorted(set(timing) - set(directions))
    if unknown:
        raise ValueError(f"{origin}: timing for unknown directions {unknown}")
    extension = None
    if rate_extension:
        r = rate_extension
        if r["direction"] not in directions or directions[r["direction"]].space != HIDDEN_SPACE:
            raise ValueError(
                f"{origin}: rate_extension direction {r['direction']!r} is not a hidden direction of the bundle"
            )
        extension = RateExtension(
            direction=r["direction"],
            row_edges=tuple(float(e) for e in r["row_edges"]),
            below=tuple(float(e) for e in r["below"]),
            above=tuple(float(e) for e in r["above"]),
            doses={lang: {int(b): float(d) for b, d in doses.items()} for lang, doses in r["languages"].items()},
            reference_sides=tuple(r.get("reference_sides", ("slower", "faster"))),
        )
    release = None
    if steer_release is not None:
        r = steer_release
        if not isinstance(r, dict) or set(r) - _RELEASE_KEYS or _RELEASE_KEYS - {"exempt"} - set(r):
            raise ValueError(
                f"{origin}: steer_release must be {{onset_seconds, ramp_seconds, hold, exempt?}}, got {r!r}"
            )
        try:
            check_release(float(r["onset_seconds"]), float(r["ramp_seconds"]), float(r["hold"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{origin}: steer_release: {exc}") from None
        exempt = tuple(r.get("exempt", ()))
        bad = sorted(d for d in exempt if d not in directions or directions[d].space != HIDDEN_SPACE)
        if bad:
            raise ValueError(f"{origin}: steer_release exempts {bad}, which are not hidden directions of the bundle")
        release = ReleaseSchedule(float(r["onset_seconds"]), float(r["ramp_seconds"]), float(r["hold"]), exempt)
    return SteeringBundle(
        name=name,
        path=origin,
        sets=dict(sets),
        directions=directions,
        timing=timing,
        ui=ui,
        rate_extension=extension,
        steer_release=release,
    )


_RELEASE_KEYS = {"onset_seconds", "ramp_seconds", "hold", "exempt"}


def load_bundle(path: str | Path) -> SteeringBundle:
    """A bundle file (``--steer-bundle``): its JSON and the set directories relative to it."""
    path = Path(path).expanduser()
    spec = json.loads(path.read_text())
    sets: Dict[str, DirectionSet] = {}
    for set_name, rel in spec["sets"].items():
        directory = (path.parent / rel).resolve()
        loaded = DirectionSet.load(directory)
        if loaded is None:
            raise ValueError(f"{path}: set {set_name!r} has no directions at {directory}")
        sets[set_name] = loaded
    return build_bundle(
        path,
        spec.get("name", path.stem),
        sets,
        timing=spec.get("timing", {}),
        ui=spec.get("ui", {}),
        rate_extension=spec.get("rate_extension"),
        steer_release=spec.get("steer_release"),
    )


def request_release(
    bundle: SteeringBundle | None,
    hold: float | None,
    onset_seconds: float | None,
    ramp_seconds: float | None,
    *,
    hint: str,
) -> ReleaseSchedule | None:
    """A request's steer-then-hold schedule: its ``steer_hold`` / ``steer_onset_seconds`` / ``steer_ramp_seconds``
    (None = the bundle's ``steer_release`` default) with the bundle's exempt directions. None when the request-level
    controls stay constant (hold 1: the bundle has no ``steer_release``, or the request asks for 1); a hold below 1
    needs the bundle's ``steer_release``, which names the directions that never release (``hint``: how this server
    loads a bundle)."""
    default = None if bundle is None else bundle.steer_release
    if default is None:
        if hold is None or float(hold) == 1.0:
            return None
        if bundle is None:
            raise ValueError(
                f"This server has no steering bundle ({hint}); steer_hold below 1 needs its steer_release."
            )
        raise ValueError(
            "steer_hold below 1 needs a steering bundle with steer_release; this server's bundle has none."
        )
    release = ReleaseSchedule(
        onset_seconds=default.onset_seconds if onset_seconds is None else float(onset_seconds),
        ramp_seconds=default.ramp_seconds if ramp_seconds is None else float(ramp_seconds),
        hold=default.hold if hold is None else float(hold),
        exempt=default.exempt,
    )
    check_release(release.onset_seconds, release.ramp_seconds, release.hold)
    return None if release.hold == 1.0 else release


def _duration_factor(bundle: SteeringBundle, controls: Mapping[str, float]) -> float:
    """Expected duration multiplier of a set of controls (product over the timed ones)."""
    factor = 1.0
    for did, table in bundle.timing.items():
        dose = float(controls.get(did, 0.0))
        if dose:
            xs, ys = zip(*sorted((float(x), float(y)) for x, y in table))
            factor *= float(np.interp(dose, xs, ys))
    return factor


@dataclass
class ResolvedControls:
    """What a request's controls do: the moved speaker embedding (or the original), the hidden
    delta laid out per steered source (flattened; None if no hidden control), the span
    schedule tensor [K, 3 + n_sources*hidden] (start frame, end frame, ramp frames, delta) or None,
    and the character range [start, end) of each schedule row (zonos3.tts.span_follow places
    the rows by the reading position instead). With ``release_exempt`` the request-level hidden
    controls split: the exempt ones stay in ``hidden_delta`` (constant), the others make
    ``release_delta`` (same layout, None if none; weighted per frame by ``release_weight``)."""

    speaker_embedding: torch.Tensor | None
    hidden_delta: torch.Tensor | None
    schedule: torch.Tensor | None
    span_chars: list[tuple[int, int]] | None = None
    release_delta: torch.Tensor | None = None


def check_controls(bundle: SteeringBundle, controls: Mapping[str, float], span_ids: Sequence[str] = ()) -> None:
    """Raise ValueError for an id the bundle does not serve (in ``controls`` or ``span_ids``)
    or a non-finite dose in ``controls``."""
    for did in list(controls) + list(span_ids):
        if did not in bundle.directions:
            raise ValueError(f"Unknown control {did!r}; see /v2/controls.")
    for did, v in controls.items():
        if not np.isfinite(v):
            raise ValueError(f"Control {did!r} must be finite, got {v!r}.")


def resolve_controls(
    bundle: SteeringBundle,
    controls: Mapping[str, float] | None,
    spans: Sequence[Mapping] | None,
    *,
    text: str,
    speaker_embedding: torch.Tensor | None,
    sources: Sequence[int],
    hidden_size: int,
    frames_per_char: float | None = None,
    release_exempt: Collection[str] | None = None,
) -> ResolvedControls:
    controls = {k: float(v) for k, v in (controls or {}).items() if float(v) != 0.0}
    spans = list(spans or [])
    check_controls(bundle, controls, [d for s in spans for d in s.get("controls", {})])

    def hidden_vector(doses: Mapping[str, float]) -> torch.Tensor | None:
        parts = [
            (bundle.directions[d], v)
            for d, v in doses.items()
            if float(v) != 0.0 and bundle.directions[d].space == HIDDEN_SPACE
        ]
        if not parts:
            return None
        out = torch.zeros(len(sources), hidden_size, dtype=torch.float32)
        for d, v in parts:
            if d.vector.numel() != hidden_size:
                raise ValueError(
                    f"Control {d.id!r} has dim {d.vector.numel()}, the model's hidden size is {hidden_size}."
                )
            out[list(sources).index(int(d.source_index))] += float(v) * d.vector
        return out.flatten()

    # speaker embedding: every raw control at once, norm kept
    raw = {d: v for d, v in controls.items() if bundle.directions[d].space == RAW_SPACE}
    embedding = speaker_embedding
    if raw:
        if speaker_embedding is None:
            names = ", ".join(sorted(raw))
            raise ValueError(f"Controls {names} move the speaker embedding and need speaker_mode=embedding.")
        base = speaker_embedding.detach().to(torch.float32, copy=True).reshape(-1).cpu()
        delta = torch.zeros_like(base)
        for did, v in raw.items():
            vec = bundle.directions[did].vector
            if vec.numel() != base.numel():
                raise ValueError(f"Control {did!r} has dim {vec.numel()}, the speaker embedding has {base.numel()}.")
            delta += v * vec
        moved = base + delta
        embedding = (
            moved * (torch.linalg.vector_norm(base) / torch.linalg.vector_norm(moved).clamp_min(1e-8))
        ).contiguous()

    release = None
    if release_exempt is None:
        hidden = hidden_vector(controls)
    else:
        hidden = hidden_vector({d: v for d, v in controls.items() if d in release_exempt})
        release = hidden_vector({d: v for d, v in controls.items() if d not in release_exempt})

    schedule, span_chars = None, None
    if spans:
        n = len(text)
        rate = float(frames_per_char or DEFAULT_FRAMES_PER_CHAR) * _duration_factor(bundle, controls)
        if rate <= 0:
            raise ValueError("control_span_rate must be positive.")
        cleaned = []
        for s in spans:
            start, end = int(s["start"]), int(s["end"])
            doses = {k: float(v) for k, v in s.get("controls", {}).items() if float(v) != 0.0}
            if not 0 <= start < end <= n:
                raise ValueError(f"control span [{start}, {end}) is outside the text (0..{n}).")
            for did in doses:
                if bundle.directions[did].space != HIDDEN_SPACE:
                    raise ValueError(
                        f"Control {did!r} moves the speaker embedding and cannot vary inside a request; "
                        "only hidden-space controls take spans."
                    )
            cleaned.append((start, end, doses))
        cleaned.sort()
        for (s0, e0, _), (s1, _e1, _) in zip(cleaned, cleaned[1:]):
            if s1 < e0:
                raise ValueError(f"control spans [{s0}, {e0}) and [{s1}, ...) overlap.")
        # frame position of every character boundary: span characters run at their own pace
        per_char = np.full(n, rate)
        for start, end, doses in cleaned:
            per_char[start:end] = rate * _duration_factor(bundle, doses)
        frame_at = np.concatenate([[0.0], np.cumsum(per_char)])
        rows, chars = [], []
        for start, end, doses in cleaned:
            vec = hidden_vector(doses)
            if vec is None:
                continue
            head = torch.tensor([round(frame_at[start]), round(frame_at[end]), RAMP_FRAMES], dtype=torch.float32)
            rows.append(torch.cat([head, vec]))
            chars.append((start, end))
        schedule = torch.stack(rows) if rows else None
        span_chars = chars if rows else None
    return ResolvedControls(
        speaker_embedding=embedding,
        hidden_delta=hidden,
        schedule=schedule,
        span_chars=span_chars,
        release_delta=release,
    )


def release_weight(frame: int, onset: int, ramp: int, hold: float) -> float:
    """Weight of the released controls at generated frame ``frame``: 1 before ``onset``, a linear ramp to ``hold``
    reaching it ``ramp`` frames after onset (a step at onset for ramp 0), ``hold`` after."""
    if frame < onset:
        return 1.0
    if frame >= onset + ramp:
        return float(hold)
    return 1.0 + (float(hold) - 1.0) * (frame - onset) / ramp


def released_delta(base: torch.Tensor | None, release_delta: torch.Tensor | None, release, frame: int):
    """``base`` plus the released controls at generated frame ``frame`` (``release``: (onset frames, ramp frames,
    hold)); ``base`` unchanged without a released delta."""
    if release_delta is None:
        return base
    w = release_weight(frame, *release)
    released = release_delta if w == 1.0 else w * release_delta
    return released if base is None else base + released


def schedule_delta(schedule: torch.Tensor | None, frame: int) -> torch.Tensor | None:
    """Sum of the span deltas active at generated frame ``frame``: weight 1 inside [start, end), ramping
    linearly to 0 over ``ramp`` frames on both sides."""
    if schedule is None:
        return None
    start, end, ramp = schedule[:, 0], schedule[:, 1], schedule[:, 2].clamp_min(1.0)
    f = float(frame)
    w = torch.minimum((f - (start - ramp)) / ramp, ((end + ramp) - f) / ramp).clamp(0.0, 1.0)
    if not bool((w > 0).any()):
        return None
    return (w[:, None] * schedule[:, 3:]).sum(0)


def controls_document(bundle: SteeringBundle, *, speaker_embedding_dim: int | None) -> dict:
    """The /v2/controls payload: sets (with their calibration), directions (raw vectors included for client-side
    geometry) and UI metadata."""
    sets = {}
    for name, d in bundle.sets.items():
        entry = {"space": d.space, "modes": ["embedding"] if d.space == RAW_SPACE else ["embedding", "reference"]}
        if d.space == HIDDEN_SPACE:
            entry["source_index"] = int(d.source_index)
        if d.calibration is not None:
            entry["calibration"] = d.calibration.document()
        sets[name] = entry
    directions = {}
    for did, d in bundle.directions.items():
        kind = "axis" if d.name in bundle.sets[d.set].axes else "named"
        entry = {"set": d.set, "space": d.space, "kind": kind, "norm": float(torch.linalg.vector_norm(d.vector))}
        if d.space == RAW_SPACE:
            entry["vector"] = [round(float(x), 6) for x in d.vector]
        directions[did] = entry
    return {
        "bundle": bundle.name,
        "embedding_dim": speaker_embedding_dim,
        "sets": sets,
        "directions": directions,
        "timing": bundle.timing,
        "ui": bundle.ui,
        "steer_release": release_document(bundle),
    }


def release_document(bundle: SteeringBundle | None) -> dict | None:
    """The server defaults of steer_hold / steer_onset_seconds / steer_ramp_seconds and the exempt directions (the
    bundle's ``steer_release``), or None: every request-level control is constant (/v2/controls, /v2/capabilities)."""
    r = None if bundle is None else bundle.steer_release
    if r is None:
        return None
    return {"onset_seconds": r.onset_seconds, "ramp_seconds": r.ramp_seconds, "hold": r.hold, "exempt": list(r.exempt)}


def disabled_controls(bundle: SteeringBundle | None, speaker_id: str | None) -> list[str]:
    """Control ids a set calibration gives this speaker dose 0 (a speaker object's ``disabled_controls``):
    ``<set>.<name>``, or ``<set>.<name>-`` when only the negative pole is disabled; bundle direction order."""
    if bundle is None:
        return []
    return [
        f"{name}.{c}"
        for name, d in bundle.sets.items()
        if d.calibration is not None
        for c in d.calibration.disabled(speaker_id, d.names)
    ]
