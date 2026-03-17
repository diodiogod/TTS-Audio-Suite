#!/usr/bin/env python3
"""
TTS Audio Suite - Auxiliary Model Docs Generator
Generates documentation for helper/post-process model registries from YAML source of truth.
"""

from pathlib import Path
import yaml


def load_data():
    yaml_path = Path(__file__).parent.parent / "docs/Dev reports/tts_audio_suite_aux_models.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_if_changed(file_path: Path, new_content: str) -> bool:
    if file_path.exists():
        existing = file_path.read_text(encoding="utf-8")
        if existing == new_content:
            return False
    file_path.write_text(new_content, encoding="utf-8")
    return True


def generate_aux_model_sources(data):
    categories = data.get("categories", [])

    output = []
    output.append("# Auxiliary Model Sources")
    output.append("")
    output.append("Helper/post-process models used by standalone utility nodes.")
    output.append("These are not engines and are documented separately to avoid polluting the engine registry.")
    output.append("")

    for category in categories:
        output.append(f"## {category['name']}")
        output.append("")
        output.append(category.get("description", ""))
        output.append("")
        if category.get("node_targets"):
            output.append(f"**Used by:** {', '.join(category['node_targets'])}")
            output.append("")
        output.append("| Model | Repo | Size | Auto-Download | Languages | Notes |")
        output.append("|---|---|---|---|---|---|")

        for model in category.get("models", []):
            name = model.get("name", "-")
            repo_id = model.get("repo_id", "")
            repo_url = f"https://huggingface.co/{repo_id}" if repo_id else ""
            repo_display = f"[{repo_id}]({repo_url})" if repo_id else "-"
            size = model.get("size", "-")
            auto_download = "✅" if model.get("auto_download") else "❌"
            languages = ", ".join(model.get("languages", [])) or "-"
            notes = model.get("notes", "")
            output.append(f"| {name} | {repo_display} | {size} | {auto_download} | {languages} | {notes} |")

        output.append("")

    output.append("*Generated from [tts_audio_suite_aux_models.yaml](Dev%20reports/tts_audio_suite_aux_models.yaml).*")
    output.append("")
    return "\n".join(output)


def generate_aux_model_layouts(data):
    layouts = data.get("model_layouts_markdown", "")
    if layouts:
        return layouts.strip() + "\n"

    output = []
    output.append("# Auxiliary Model Layouts")
    output.append("")
    output.append("No layout data found in YAML source.")
    output.append("")
    return "\n".join(output)


def main():
    data = load_data()
    docs_dir = Path(__file__).parent.parent / "docs"

    print("Generating Auxiliary Model Sources...")
    sources = generate_aux_model_sources(data)
    if write_if_changed(docs_dir / "AUX_MODEL_SOURCES.md", sources):
        print(f"✅ Written to {docs_dir / 'AUX_MODEL_SOURCES.md'}")
    else:
        print(f"⏭️  Skipped {docs_dir / 'AUX_MODEL_SOURCES.md'} (unchanged)")

    print("Generating Auxiliary Model Layouts...")
    layouts = generate_aux_model_layouts(data)
    if write_if_changed(docs_dir / "AUX_MODEL_LAYOUTS.md", layouts):
        print(f"✅ Written to {docs_dir / 'AUX_MODEL_LAYOUTS.md'}")
    else:
        print(f"⏭️  Skipped {docs_dir / 'AUX_MODEL_LAYOUTS.md'} (unchanged)")

    print("✅ Auxiliary model docs generated successfully!")


if __name__ == "__main__":
    main()
