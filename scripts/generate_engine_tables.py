#!/usr/bin/env python3
"""
TTS Audio Suite - Engine Tables Generator
Generates three documentation tables from YAML source of truth
"""

import yaml
from pathlib import Path


def load_data():
    """Load YAML data"""
    yaml_path = Path(__file__).parent.parent / "docs/Dev reports/tts_audio_suite_engines.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_support(value, notes=""):
    """Format boolean/partial support with optional notes"""
    if value is None:
        return "N/A"
    elif value == "partial":
        return f"⚠️ {notes}" if notes else "⚠️"
    elif value is True:
        return f"✅ {notes}" if notes else "✅"
    else:
        return "❌"


def generate_engine_comparison(data):
    """Generate main engine comparison table"""
    engines = data["engines"]

    output = []
    output.append("# TTS Engines Reference Tables")
    output.append("")
    output.append("## Engine Comparison")
    output.append("")
    output.append("| Engine             | Models                                    | Size         | TTS | SRT | VC  | Special Features                                                                         | Languages                                                                                |")
    output.append("| ------------------ | ----------------------------------------- | ------------ | --- | --- | --- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |")

    for e in engines:
        # Extract flags from languages
        flags = "".join(
            lang_data["flag"]
            for lang_data in e["languages"].values()
            if lang_data["supported"]
        )

        # Format special features with proper spacing
        features = ", ".join(e.get("special_features", []))

        # Handle ChatterBox 23L special case for languages display
        if e["id"] == "chatterbox-23l":
            flags += "(+9)"

        row = [
            f"**{e['name']}**".ljust(18),
            e["models"].ljust(41),
            e["size"].ljust(12),
            format_support(e["capabilities"]["tts"]),
            format_support(e["capabilities"]["srt"]),
            format_support(e["capabilities"]["vc"]),
            features.ljust(88),
            flags.ljust(88)
        ]

        output.append("| " + " | ".join(row) + " |")

    return "\n".join(output)


def generate_language_support(data):
    """Generate language support matrix"""
    engines = data["engines"]
    lang_meta = data["language_metadata"]

    # Get all language codes in order
    lang_codes = list(lang_meta.keys())

    output = []
    output.append("# Language Support by Engine")
    output.append("")
    output.append("## Language Support by Engine")
    output.append("")

    # Build header
    header = "| Language       | Code |"
    separator = "|---|---|"
    for e in engines:
        header += f" {e['name']} |"
        separator += "---|"

    output.append(header)
    output.append(separator)

    # Build rows for each language
    for lang_code in lang_codes:
        lang_info = lang_meta[lang_code]
        # Get flag from first engine's language data (all have same flag)
        flag = engines[0]["languages"][lang_code]["flag"]

        row = [
            f"{flag} **{lang_info['name']}**".ljust(14),
            lang_info['code'].ljust(4)
        ]

        for e in engines:
            lang_support = e["languages"][lang_code]
            cell = format_support(lang_support["supported"], lang_support["notes"])
            row.append(cell)

        output.append("| " + " | ".join(row) + " |")

    # Add notes
    output.append("")
    output.append("**Notes:**")
    output.append("")
    for note in data["table_notes"]["language_support"]:
        output.append(f"- {note}")

    return "\n".join(output)


def generate_feature_comparison(data):
    """Generate feature comparison matrix"""
    engines = data["engines"]
    feat_meta = data["feature_metadata"]

    output = []
    output.append("# Feature Comparison Matrix")
    output.append("")
    output.append("## Feature Comparison Matrix")
    output.append("")

    # Build header
    header = "| Feature                      |"
    separator = "|---|"
    for e in engines:
        header += f" {e['name']} |"
        separator += "---|"

    output.append(header)
    output.append(separator)

    # Build rows for each feature
    for feat_key, feat_info in feat_meta.items():
        row = [feat_info["display"].ljust(28)]

        for e in engines:
            feat_support = e["features"][feat_key]
            cell = format_support(feat_support["supported"], feat_support["notes"])
            row.append(cell)

        output.append("| " + " | ".join(row) + " |")

    return "\n".join(output)


def main():
    """Generate all three tables"""
    data = load_data()
    docs_dir = Path(__file__).parent.parent / "docs"

    # Generate Engine Comparison
    print("Generating Engine Comparison table...")
    engine_comp = generate_engine_comparison(data)
    with open(docs_dir / "ENGINE_COMPARISON.md", "w", encoding="utf-8") as f:
        f.write(engine_comp)
    print(f"✅ Written to {docs_dir / 'ENGINE_COMPARISON.md'}")

    # Generate Language Support
    print("Generating Language Support table...")
    lang_support = generate_language_support(data)
    with open(docs_dir / "LANGUAGE_SUPPORT.md", "w", encoding="utf-8") as f:
        f.write(lang_support)
    print(f"✅ Written to {docs_dir / 'LANGUAGE_SUPPORT.md'}")

    # Generate Feature Comparison
    print("Generating Feature Comparison table...")
    feat_comp = generate_feature_comparison(data)
    with open(docs_dir / "FEATURE_COMPARISON.md", "w", encoding="utf-8") as f:
        f.write(feat_comp)
    print(f"✅ Written to {docs_dir / 'FEATURE_COMPARISON.md'}")

    print("\n✅ All tables generated successfully!")


if __name__ == "__main__":
    main()
