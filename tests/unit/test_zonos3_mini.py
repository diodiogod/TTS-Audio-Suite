import base64
import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engines.zonos3_mini import ensure_vendor_path
from engines.zonos3_mini import zonos3_mini_downloader as downloader_module
from engines.zonos3_mini.languages import LANGUAGES, to_suite_language, to_zonos_language
from engines.zonos3_mini.steering_controls import DELIVERY, VOICE_DESIGN, build_controls, parse_extra_controls
from engines.zonos3_mini.zonos3_mini_downloader import FILES, Zonos3MiniDownloader
from utils.text.chunking import ImprovedChatterBoxChunker


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER_MODULE = _load("zonos3_mini_adapter_test_module", "engines/adapters/zonos3_mini_adapter.py")
PROCESSOR_MODULE = _load("zonos3_mini_processor_test_module", "nodes/zonos3_mini/zonos3_mini_processor.py")
SRT_MODULE = _load("zonos3_mini_srt_processor_test_module", "nodes/zonos3_mini/zonos3_mini_srt_processor.py")


class StubEngine:
    """Stands in for Zonos3MiniEngine: records the native calls an adapter makes."""

    def __init__(self):
        self.requests = []
        self.embedded = []
        self.encoded = []

    def _ensure_model_loaded(self, decoder="keep2_plain"):
        pass

    def preset(self, name):
        return {
            "codes": np.full((4, 10), len(name), dtype=np.int32),
            "embedding": np.full(192, len(name), dtype=np.float32),
        }

    def embed_speaker(self, waveform, sample_rate):
        self.embedded.append((waveform.clone(), sample_rate))
        return torch.full((192,), float(waveform.sum()))

    def encode_reference(self, waveform, sample_rate):
        self.encoded.append((waveform.clone(), sample_rate))
        return torch.arange(30, dtype=torch.int32).reshape(3, 10)

    def generate(self, request):
        self.requests.append(request)
        return torch.full((1, 44100), 0.25), None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(ADAPTER_MODULE.Zonos3MiniEngineAdapter, "_ensure_model_loaded", lambda self: None)
    engine = StubEngine()
    monkeypatch.setattr(ADAPTER_MODULE.Zonos3MiniEngineAdapter, "_get_engine", lambda self: engine)
    instance = ADAPTER_MODULE.Zonos3MiniEngineAdapter({"language": "English (US)"})
    instance.audio_cache.clear_cache()
    instance.engine = engine
    return instance


@pytest.mark.unit
def test_release_paths_resolve_inside_the_local_model_root(tmp_path):
    ensure_vendor_path()
    hub_loaded_before = "huggingface_hub" in sys.modules
    from zonos3.utils import release

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.pt").write_bytes(b"x")
    release.set_local_root(str(tmp_path))
    try:
        assert release.resolve_file("hf://Zyphra/ZONOS3-Mini/a/b.pt") == str(tmp_path / "a" / "b.pt")
        assert release.resolve_dir_or_local("hf://Zyphra/ZONOS3-Mini/a") == str(tmp_path / "a")
        with pytest.raises(FileNotFoundError, match="ZONOS3-Mini model file missing"):
            release.resolve_file("hf://Zyphra/ZONOS3-Mini/a/absent.pt")
        with pytest.raises(ValueError, match="unexpected model repository Other/Repo"):
            release.resolve_file("hf://Other/Repo/x")
        release.set_local_root(None)
        with pytest.raises(RuntimeError, match="local model root is not set"):
            release.resolve_file("hf://Zyphra/ZONOS3-Mini/a/b.pt")
    finally:
        release.set_local_root(None)
    assert ("huggingface_hub" in sys.modules) == hub_loaded_before


@pytest.mark.unit
def test_downloader_reports_wrong_sizes_and_never_downloads_local_models(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "FILES", {"base": [("lm/params.json", 4), ("lm/model.pth", 6)]})
    root = tmp_path / "zonos3_mini" / "Broken"
    (root / "lm").mkdir(parents=True)
    (root / "lm" / "params.json").write_bytes(b"1234")
    (root / "lm" / "model.pth").write_bytes(b"12")  # truncated

    assert Zonos3MiniDownloader.missing(str(root), ["base"]) == ["lm/model.pth"]

    monkeypatch.setattr(Zonos3MiniDownloader, "model_root", lambda self, model_identifier="ZONOS3-Mini": str(root))

    def fail_download(self, root, rel_paths):
        raise AssertionError("local models must not be downloaded")

    monkeypatch.setattr(Zonos3MiniDownloader, "_download", fail_download)
    with pytest.raises(FileNotFoundError, match="local model 'Broken' is missing: lm/model.pth"):
        Zonos3MiniDownloader().ensure("local:Broken", ["base"])

    (root / "lm" / "model.pth").write_bytes(b"123456")
    assert Zonos3MiniDownloader().ensure("local:Broken", ["base"]) == str(root)


@pytest.mark.unit
def test_language_codes_round_trip_and_aliases_resolve():
    assert len(LANGUAGES) == 44
    for display, code in LANGUAGES.items():
        assert to_zonos_language(code) == code
        assert to_zonos_language(display) == code
        assert to_zonos_language(to_suite_language(code)) == code

    assert to_zonos_language("en") == "en_us"
    assert to_zonos_language("EN-GB") == "en_gb"
    assert to_zonos_language("fr") == "fr_fr"
    assert to_zonos_language("pt-pt") == "pt_br"
    assert to_zonos_language("zh-tw") == "cmn"
    assert to_zonos_language("nb") == "no"
    assert to_zonos_language("English (UK)") == "en_gb"
    with pytest.raises(ValueError, match="does not support language 'klingon'"):
        to_zonos_language("klingon")


@pytest.mark.unit
def test_node_languages_match_the_vendored_registry():
    ensure_vendor_path()
    from zonos3 import languages

    assert list(LANGUAGES.values()) == [language.code for language in languages.SERVER]


@pytest.mark.unit
def test_shared_parser_accepts_zonos_language_tags():
    from utils.text.character_parser.base_parser import CharacterParser

    parser = CharacterParser()
    parser.set_available_characters(["zorblax", "narrator"])

    def parse(text):
        segments = parser.parse_text_segments(text, engine_type="zonos3_mini")
        assert len(segments) == 1
        return segments[0]

    assert parse("[uk] Привіт").language == "uk"
    assert parse("[th] สวัสดี").language == "th"
    assert parse("[ka] გამარჯობა").language == "ka"

    tagged = parse("[te:Zorblax|seed:42] నమస్కారం")
    assert (tagged.language, tagged.character.lower(), tagged.parameters) == ("te", "zorblax", {"seed": 42})

    italian = parse("[it] ciao")
    assert "[it]" in italian.text and italian.character == "narrator"


@pytest.mark.unit
def test_steering_sliders_map_to_native_directions():
    assert build_controls(DELIVERY, {"projection": -2.0}) == {"delivery.whisper": 2.0}
    assert build_controls(DELIVERY, {"projection": 1.5, "happy": 0.0}) == {"delivery.loud": 1.5}
    assert build_controls(DELIVERY, {"calm": 4.0}) == {"emotions.serene": 4.0}
    assert build_controls(VOICE_DESIGN, {"mandarin": 1.5, "gender": 0.0}) == {
        "mandarin.cmn": 1.5,
        "mandarin_hidden.cmn": 0.5,
    }
    assert build_controls(VOICE_DESIGN, {"bright": 2.0, "hindi": 1.0}) == {
        "texture.shrill": 2.0,
        "accents_l1.hin": 1.0,
    }

    assert parse_extra_controls("emotions.proud=3\n\n mood.dominance = -1.5 ") == {
        "emotions.proud": 3.0,
        "mood.dominance": -1.5,
    }
    for bad in ("bad line", "proud=3", "emotions.proud=lots"):
        with pytest.raises(ValueError, match="must be 'set.direction=number'"):
            parse_extra_controls(bad)


@pytest.mark.unit
def test_every_slider_direction_exists_in_the_bundled_steering_assets():
    import json

    steering = REPO_ROOT / "engines" / "zonos3_mini" / "assets" / "steering"
    sets = json.loads((steering / "studio_bundle_mect60k_x10.json").read_text())["sets"]
    for table in (VOICE_DESIGN, DELIVERY):
        for widget, low, high, mapping, _tooltip in table:
            for value in (low, high):
                for direction in mapping(value) if value else {}:
                    set_name, name = direction.split(".")
                    assert (steering / sets[set_name] / f"{name}.npy").is_file(), (widget, direction)


@pytest.mark.unit
def test_delivery_node_merges_extra_controls_and_drops_neutral_sliders():
    steering_nodes = _load("zonos3_mini_steering_nodes_test_module", "nodes/engines/zonos3_mini_steering_nodes.py")

    (delivery,) = steering_nodes.Zonos3MiniDeliveryNode().create_steering(
        happy=3.0, sad=0.0, extra_controls="emotions.proud=3", steer_hold=0.5
    )
    assert delivery["controls"] == {"emotions.happy": 3.0, "emotions.proud": 3.0}
    assert delivery["steer_hold"] == 0.5 and delivery["steer_onset_seconds"] == -1.0

    (design,) = steering_nodes.Zonos3MiniVoiceDesignNode().create_steering(gender=-2.0, age=0.0)
    assert design == {"controls": {"identity.gender": -2.0}}


@pytest.mark.unit
def test_adapter_embeds_the_connected_tensor_without_a_transcript(adapter):
    waveform = torch.linspace(-0.5, 0.5, 22050).reshape(1, 1, -1)
    voice_ref = {"audio": {"waveform": waveform, "sample_rate": 22050}, "audio_path": "/stale/voice.wav"}

    audio = adapter.generate_single("Hello there.", voice_ref, seed=7)

    assert audio.shape == (1, 44100)
    (embedded_waveform, embedded_rate), = adapter.engine.embedded
    assert embedded_rate == 22050 and torch.equal(embedded_waveform, waveform[0])
    (request,) = adapter.engine.requests
    assert request["speaker_mode"] == "embedding" and "speaker_reference_codes" not in request
    vector = np.load(io.BytesIO(base64.b64decode(request["speaker_embedding_base64"])), allow_pickle=False)
    assert vector.shape == (192,) and vector.dtype == np.float32
    assert request["language"] == "en_us" and request["seed"] == 7


@pytest.mark.unit
def test_adapter_reference_mode_sends_codes_and_no_embedding(adapter):
    adapter.update_config({"language": "French", "voice_mode": "reference", "decode_context": True})
    voice_ref = {"audio": {"waveform": torch.ones(1, 1, 44100) * 0.1, "sample_rate": 44100}}

    adapter.generate_single("Bonjour.", voice_ref, seed=1)
    adapter.generate_single("Encore.", voice_ref, seed=1)

    assert len(adapter.engine.encoded) == 1  # the clip is encoded once and reused
    first, second = adapter.engine.requests
    assert first["speaker_mode"] == "reference" and "speaker_embedding_base64" not in first
    assert first["speaker_reference_codes"] == torch.arange(30).reshape(3, 10).tolist()
    assert first["decode_context"] is True and first["language"] == "fr_fr"
    assert second["speaker_reference_codes"] == first["speaker_reference_codes"]


@pytest.mark.unit
def test_adapter_uses_the_preset_voice_without_reference_audio(adapter):
    adapter.update_config({"language": "English (US)", "voice_preset": "Wyatt"})
    adapter.generate_single("Hello.", None, seed=1)
    (request,) = adapter.engine.requests
    vector = np.load(io.BytesIO(base64.b64decode(request["speaker_embedding_base64"])), allow_pickle=False)
    assert np.array_equal(vector, np.full(192, len("Wyatt"), dtype=np.float32))
    assert not adapter.engine.embedded


@pytest.mark.unit
def test_adapter_cache_hits_identical_requests_and_misses_on_any_change(adapter):
    base = {"language": "English (US)", "voice_preset": "Savannah", "controls": {"emotions.happy": 2.0}}
    voice = {"audio": {"waveform": torch.zeros(1, 1, 44100), "sample_rate": 44100}}

    def calls_after(config, voice_ref=voice, seed=5):
        adapter.update_config(config)
        adapter.generate_single("Same text.", voice_ref, seed=seed)
        return len(adapter.engine.requests)

    assert calls_after(base) == 1
    assert calls_after(base) == 1  # identical request: served from the cache
    assert calls_after({**base, "speaking_rate_bucket": 12}) == 2
    assert calls_after({**base, "controls": {"emotions.happy": 3.0}}) == 3
    assert calls_after(base, seed=6) == 4
    louder = {"audio": {"waveform": torch.ones(1, 1, 44100) * 0.2, "sample_rate": 44100}}
    assert calls_after(base, voice_ref=louder) == 5
    assert calls_after(base, voice_ref=None) == 6
    assert calls_after({**base, "voice_preset": "Wyatt"}, voice_ref=None) == 7
    assert calls_after(base) == 7  # the first request is still cached


@pytest.mark.unit
def test_adapter_request_uses_only_native_generate_request_fields(adapter):
    ensure_vendor_path()
    from zonos3.tts.plan import GENERATE_DEFAULTS

    adapter.update_config(
        {
            "language": "German",
            "voice_mode": "reference",
            "max_new_frames": 300,
            "controls": {"delivery.tempo": 1.0, "emotions.sad": 0.0},
            "steer_hold": 0.5,
            "steer_onset_seconds": -1.0,
            "steer_ramp_seconds": 2.0,
            "laci": True,
        }
    )
    adapter.generate_single("Guten Tag.", None, seed=3, enable_audio_cache=False)
    (request,) = adapter.engine.requests

    assert set(request) <= set(GENERATE_DEFAULTS)
    assert request["max_tokens"] == 300 and request["topk"] == 80 and request["laci"] is True
    assert request["controls"] == {"delivery.tempo": 1.0}
    assert request["steer_hold"] == 0.5 and request["steer_ramp_seconds"] == 2.0
    assert "steer_onset_seconds" not in request
    assert request["condition_buckets"] == {"loudness": 10, "fmax": 10, "speaking_rate": 30}


@pytest.mark.unit
def test_adapter_does_not_cache_failed_generations(adapter):
    def fail(request):
        raise InterruptedError("stop")

    adapter.engine.generate, original = fail, adapter.engine.generate
    with pytest.raises(InterruptedError):
        adapter.generate_single("Cancelled line.", None, seed=9)
    adapter.engine.generate = original
    adapter.generate_single("Cancelled line.", None, seed=9)
    assert len(adapter.engine.requests) == 1


class RecordingAdapter:
    def __init__(self, seconds=1.0):
        self.seconds = seconds
        self.calls = []
        self.config = {}

    def update_config(self, config):
        self.config = dict(config)

    def generate_single(self, text, voice_ref, seed=0, enable_audio_cache=True, character_name=None):
        self.calls.append({"text": text, "seed": seed, "config": dict(self.config), "voice_ref": voice_ref})
        return torch.full((1, int(self.seconds * 44100)), 0.5)


@pytest.fixture
def no_interrupt(monkeypatch):
    for module in (PROCESSOR_MODULE, SRT_MODULE):
        monkeypatch.setattr(module.model_management, "interrupt_processing", False, raising=False)


@pytest.mark.unit
def test_processor_turns_pause_tags_into_exact_silence(no_interrupt):
    recording = RecordingAdapter()
    processor = PROCESSOR_MODULE.Zonos3MiniProcessor(recording, {"language": "English (US)"})

    records = processor.process_text("Hello [pause:0.25] again.", {}, seed=1, show_text_logging=False)

    assert [record["text"] for record in records] == ["Hello", "[pause:0.250s]", "again."]
    assert records[1]["waveform"].shape[-1] == 11025 and not records[1]["waveform"].any()
    assert [call["text"] for call in recording.calls] == ["Hello", "again."]


@pytest.mark.unit
@pytest.mark.parametrize("method", ["auto", "concatenate", "silence_padding", "crossfade"])
@pytest.mark.parametrize("duration", [0.05, 0.25])
def test_processor_preserves_pause_duration_when_combining(no_interrupt, method, duration):
    processor = PROCESSOR_MODULE.Zonos3MiniProcessor(RecordingAdapter(), {"language": "English (US)"})
    text = f"Alpha [pause:{duration}] omega."
    records = processor.process_text(text, {}, seed=1, show_text_logging=False)

    audio, info = processor.combine_audio_segments(records, method=method, original_text=text, return_info=True)

    expected = torch.cat([record["waveform"] for record in records], dim=-1)
    assert torch.equal(audio, expected)
    assert info["junction_methods"] == ["concatenate", "concatenate"]
    assert info["chunk_timings"][-1]["end"] == pytest.approx(audio.shape[-1] / 44100)


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "A short opening. " + "word " * 180 + "End.",
    "First clause, " + "word " * 180 + "end.",
    "a, " * 220 + "end.",
    "word " * 180 + "End.",
], ids=["long_sentence_after_short", "long_comma_part", "comma_punctuation", "long_sentence_alone"])
def test_shared_chunker_enforces_limit_without_changing_text(text):
    chunks = ImprovedChatterBoxChunker.split_into_chunks(text, 400)

    assert chunks and all(0 < len(chunk) <= 400 for chunk in chunks)
    assert " ".join(chunks) == " ".join(text.split())


@pytest.mark.unit
def test_shared_chunker_splits_single_oversized_token():
    text = "Opening. " + "x" * 905
    chunks = ImprovedChatterBoxChunker.split_into_chunks(text, 400)

    assert all(0 < len(chunk) <= 400 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


@pytest.mark.unit
def test_processor_accepts_long_form_character_budget(no_interrupt):
    recording = RecordingAdapter(seconds=0.001)
    processor = PROCESSOR_MODULE.Zonos3MiniProcessor(recording, {"language": "English (US)"})
    text = "A complete sentence. " * 700
    budget = ImprovedChatterBoxChunker.validate_chunking_params(20_000)

    processor.process_text(text, {}, seed=1, max_chars_per_chunk=budget, show_text_logging=False)

    assert budget == 20_000
    assert [call["text"] for call in recording.calls] == [text.strip()]


@pytest.mark.unit
def test_processor_applies_language_tags_segment_parameters_and_chunking(no_interrupt):
    recording = RecordingAdapter()
    processor = PROCESSOR_MODULE.Zonos3MiniProcessor(
        recording, {"language": "English (US)", "speaking_rate_bucket": 30, "temperature": 0.75}
    )
    narrator = {"audio": {"waveform": torch.zeros(1, 1, 8), "sample_rate": 44100}}

    processor.process_text(
        "Plain line. [fr:] Bonjour. [Zorblax|seed:42|temperature:0.5|rate_bucket:12] Tuned line.",
        {"narrator": narrator},
        seed=1,
        show_text_logging=False,
    )
    plain, french, tuned = recording.calls
    assert to_zonos_language(plain["config"]["language"]) == "en_us" and plain["seed"] == 1
    assert french["config"]["language"] == "fr_fr"
    assert (tuned["seed"], tuned["config"]["temperature"], tuned["config"]["speaking_rate_bucket"]) == (42, 0.5, 12)
    # an unknown character falls back to the narrator's reference voice
    assert tuned["voice_ref"]["audio"] is narrator["audio"]
    assert recording.config["speaking_rate_bucket"] == 30  # per-segment overrides do not leak

    recording.calls.clear()
    long_text = " ".join(f"Sentence number {index} is here." for index in range(40))
    processor.process_text(long_text, {}, seed=1, max_chars_per_chunk=200, show_text_logging=False)
    assert len(recording.calls) > 1 and all(len(call["text"]) <= 200 for call in recording.calls)

    recording.calls.clear()
    processor.process_text(long_text, {}, seed=1, enable_chunking=False, show_text_logging=False)
    assert len(recording.calls) == 1


TWO_CUE_SRT = """1
00:00:00,000 --> 00:00:02,000
Hello.

2
00:00:03,000 --> 00:00:06,000
[fr:] Bonjour.
"""


@pytest.mark.unit
def test_srt_processor_places_cues_on_the_subtitle_timeline(no_interrupt, monkeypatch):
    recording = RecordingAdapter(seconds=1.0)
    monkeypatch.setattr(SRT_MODULE, "Zonos3MiniEngineAdapter", lambda config: recording)
    processor = SRT_MODULE.Zonos3MiniSRTProcessor(None, {"language": "English (US)"})
    assert processor.srt_available

    audio, info, _report, adjusted_srt = processor.process_srt_content(
        TWO_CUE_SRT, {}, seed=10, timing_mode="pad_with_silence", timing_params={}
    )

    waveform = audio["waveform"]
    assert audio["sample_rate"] == 44100 and waveform.dim() == 3 and waveform.shape[:2] == (1, 1)
    # the timeline runs to the end of the last cue (6.0 s); cue 2 speaks from 3.0 s to 4.0 s
    assert abs(waveform.shape[-1] - 6 * 44100) <= 2940
    samples = waveform[0, 0]
    assert samples[: 44100].abs().min() > 0
    assert not samples[int(1.1 * 44100) : int(2.9 * 44100)].any()
    assert samples[int(3.05 * 44100) : int(3.95 * 44100)].abs().min() > 0
    assert not samples[int(4.1 * 44100) :].any()
    assert [call["seed"] for call in recording.calls] == [10, 11]
    assert recording.calls[1]["config"]["language"] == "fr_fr"
    assert len(processor.SRTParser().parse_srt_content(adjusted_srt)) == 2
    assert "ZONOS3-Mini" in info


# ---------------------------------------------------------------------------------------------- fast engine


@pytest.mark.unit
def test_fast_engine_falls_back_on_auto_and_raises_when_forced():
    from engines.zonos3_mini import zonos3_mini_fast
    from engines.zonos3_mini.zonos3_mini_engine import Zonos3MiniEngine

    assert "CUDA" in zonos3_mini_fast.availability("cpu", torch.float32)
    assert Zonos3MiniEngine(device="cpu", engine="auto")._fast_runner() is None
    with pytest.raises(RuntimeError, match="fast engine is not available"):
        Zonos3MiniEngine(device="cpu", engine="fast")._fast_runner()
    with pytest.raises(ValueError, match="auto, fast or reference"):
        Zonos3MiniEngine(device="cpu", engine="turbo")


def _triton_cuda():
    if not torch.cuda.is_available() or importlib.util.find_spec("triton") is None:
        pytest.skip("needs CUDA and Triton")
    ensure_vendor_path()
    return torch.device("cuda")


@pytest.mark.unit
def test_gated_delta_kernel_matches_the_reference_recurrence():
    device = _triton_cuda()
    from zonos3_sgl.kernel.triton.gated_delta import gated_delta

    generator = torch.Generator(device=device).manual_seed(0)
    rand = lambda *shape: torch.randn(*shape, device=device, generator=generator)
    heads, dim, lengths, slots = 4, 128, [5, 0, 70], [2, 0, 3]
    starts = [0, 5, 6]  # row 5 belongs to no sequence
    rows = 76
    q, k, v = rand(rows, heads, dim), torch.nn.functional.normalize(rand(rows, heads, dim), dim=-1), rand(rows, heads, dim)
    decay, beta = torch.rand(rows, heads, device=device, generator=generator), torch.rand(rows, heads, device=device, generator=generator)
    state = rand(4, heads, dim, dim) * 0.1
    expected_state, expected = state.clone(), torch.zeros(rows, heads, dim, device=device)
    for start, n, slot in zip(starts, lengths, slots):
        s = expected_state[slot].clone()
        for t in range(start, start + n):
            s = s * decay[t, :, None, None]
            delta = beta[t, :, None] * (v[t] - (s * k[t, :, None, :]).sum(-1))
            s = s + delta[:, :, None] * k[t, :, None, :]
            expected[t] = (s * (q[t] * 0.5)[:, None, :]).sum(-1)
        expected_state[slot] = s

    as_long = lambda values: torch.tensor(values, device=device)
    out = gated_delta(q, k, v, decay, beta, state, as_long(starts), as_long(lengths), as_long(slots), 0.5)

    assert torch.allclose(out, expected, atol=1e-4, rtol=1e-4)
    assert torch.allclose(state, expected_state, atol=1e-4, rtol=1e-4)
    assert torch.equal(state[1], expected_state[1])  # a slot no sequence owns is untouched


@pytest.mark.unit
def test_paged_attention_kernel_matches_causal_softmax_over_each_requests_pages():
    device = _triton_cuda()
    from zonos3_sgl.kernel.triton.paged_attention import paged_attention

    generator = torch.Generator(device=device).manual_seed(0)
    heads, kv_heads, dim, pages = 4, 2, 32, 400
    k_cache = torch.randn(pages, 1, kv_heads, dim, device=device, generator=generator)
    v_cache = torch.randn(pages, 1, kv_heads, dim, device=device, generator=generator)
    page_table = torch.randperm(pages, device=device, generator=generator)[:300].to(torch.int32).view(2, 150).contiguous()
    owner = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device=device)
    last = torch.tensor([0, 63, 64, 149], dtype=torch.int32, device=device)  # block edges of the 64-key loop
    q = torch.randn(4, heads, dim, device=device, generator=generator)

    out = paged_attention(q, k_cache, v_cache, page_table, owner, last, dim**-0.5)

    for row in range(4):
        visible = page_table[owner[row], : last[row] + 1].long()
        for head in range(heads):
            keys, values = k_cache[visible, 0, head // 2], v_cache[visible, 0, head // 2]
            weights = torch.softmax(keys @ q[row, head] * dim**-0.5, dim=0)
            assert torch.allclose(out[row, head], weights @ values, atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_gdn_layer_output_does_not_depend_on_where_prefill_ends_and_decode_starts():
    device = _triton_cuda()
    from types import SimpleNamespace

    from zonos3_sgl.models.gdn import OriginalGDN, PrefillInputs, PrefillLayout

    generator = torch.Generator(device=device).manual_seed(0)
    rand = lambda *shape: torch.randn(*shape, device=device, generator=generator)
    hidden, rows = 64, 150

    def layer():
        gdn = OriginalGDN(SimpleNamespace(hidden_size=hidden), 0)
        g = torch.Generator(device=device).manual_seed(1)
        fill = lambda t, scale: torch.randn(t.shape, device=device, generator=g) * scale
        gdn.in_proj.weight, gdn.conv.weight = fill(gdn.in_proj.weight, 0.1), fill(gdn.conv.weight, 0.5)
        gdn.out_weight, gdn.wo.weight = torch.ones(128, device=device), fill(gdn.wo.weight, 0.1)
        gdn.A_log, gdn.dt_bias = fill(gdn.A_log, 1.0), fill(gdn.dt_bias, 1.0)
        gdn.allocate_cache(3, device, torch.float32)
        return gdn

    def run(prefill_rows, slot):
        gdn = layer()
        slots = torch.tensor([slot], device=device)
        layout = PrefillLayout.build([prefill_rows // 64], device)
        padded = torch.zeros(layout.lengths[0], hidden, device=device)
        padded[:prefill_rows] = x[:prefill_rows]
        inputs = PrefillInputs(layout=layout, slots=slots, lengths=torch.tensor([prefill_rows], device=device))
        out = [gdn._forward_prefill_static(padded, inputs)[:prefill_rows]]
        gdn.enable_graph_decode(slots, OriginalGDN.static_tables(device, [1]))
        out += [gdn._forward_decode_static(x[t : t + 1], 1) for t in range(prefill_rows, rows)]
        return torch.cat(out)

    x = rand(rows, hidden)
    whole = run(rows, 0)
    assert whole.abs().max() > 1e-3
    for prefill_rows, slot in ((1, 1), (64, 2), (77, 0)):
        assert torch.allclose(run(prefill_rows, slot), whole, atol=1e-4, rtol=1e-3)
