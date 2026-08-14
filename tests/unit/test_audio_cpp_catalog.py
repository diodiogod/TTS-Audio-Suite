from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.audio_cpp.catalog import (
    AUDIO_CPP_RELEASE_VERSION,
    CatalogError,
    family_choices,
    get_model_specs_dir,
    load_catalog,
    package_choices,
    recommended_package,
    resolve_task,
)


@pytest.mark.unit
def test_pinned_release_catalog_has_exact_suite_compatible_surface():
    catalog = load_catalog()

    assert AUDIO_CPP_RELEASE_VERSION == "0.5.1"
    assert len(catalog.families) == 32
    assert len(catalog.packages) == 96
    assert set(family_choices()) == set(catalog.families)
    assert len(package_choices()) == 96
    assert set(path.name for path in get_model_specs_dir().glob("*.json")) == {
        family.spec_filename for family in catalog.families.values()
    }
    assert "vevo2" in catalog.families
    assert catalog.family("vevo2").runtime_tasks == ("tts", "vc", "s2s", "svc")
    assert catalog.family("qwen3_asr").runtime_tasks == ("asr",)
    assert catalog.family("seed_vc").runtime_tasks == ("vc", "svc")


@pytest.mark.unit
def test_catalog_merges_package_download_defaults_and_maps_local_paths():
    catalog = load_catalog()
    package = catalog.package("chatterbox_q8_0")

    assert package.repo == "audio-cpp/audio.cpp-gguf"
    assert package.revision == "main"
    assert package.local_files == (Path("chatterbox-q8_0.gguf"),)
    assert recommended_package("chatterbox") == "chatterbox_q8_0"

    # Upstream release-0.5.1 uses strip_prefix="." here.  It means no strip,
    # not a literal directory named dot.
    assert catalog.package("vietneu_tts_v3_turbo_q8_0").local_files == (Path("model.gguf"),)


@pytest.mark.unit
def test_resolve_task_uses_compiled_ids_and_specialized_package_semantics():
    assert resolve_task("chatterbox", "chatterbox_q8_0", "clone") == "clon"
    assert (
        resolve_task("qwen3_tts", "qwen3_tts_1_7b_voicedesign_q8_0", "auto") == "vdes"
    )
    assert (
        resolve_task("irodori_tts", "irodori_tts_600m_v3_voicedesign_f16", "auto") == "vdes"
    )
    assert resolve_task("qwen3_tts", "qwen3_tts_1_7b_base_q8_0", "auto") == "tts"
    assert resolve_task("pocket_tts", "pocket_tts_english_q8_0", "clone") == "tts"
    with pytest.raises(CatalogError, match="does not belong"):
        resolve_task("chatterbox", "vevo2_q8_0", "auto")


@pytest.mark.unit
def test_every_package_maps_to_safe_relative_files():
    for package in load_catalog().packages.values():
        assert package.local_files
        for path in package.local_files:
            assert not path.is_absolute()
            assert ".." not in path.parts
