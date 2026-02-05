#!/usr/bin/env python3
"""
TTS Audio Suite - Engine Tables Generator
Generates three documentation tables from YAML source of truth
Also injects condensed table into README.md between markers
"""

import yaml
import re
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


def get_speed_emoji(speed_note):
    """Convert speed note to emoji"""
    if "Very Fast" in speed_note:
        return "⚡⚡"
    elif "Fast" in speed_note:
        return "⚡"
    else:
        return "🐌"


def generate_engine_comparison(data):
    """Generate main engine comparison table"""
    engines = data["engines"]

    output = []
    output.append("# TTS Engines Reference Tables")
    output.append("")
    output.append("## Engine Comparison")
    output.append("")
    output.append("| Engine             | Models                                    | Size         | TTS | SRT | VC  | ASR | Special Features                                                                         | Languages                                                                                |")
    output.append("| ------------------ | ----------------------------------------- | ------------ | --- | --- | --- | --- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |")

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
            format_support(e["capabilities"]["asr"]),
            features.ljust(88),
            flags.ljust(88)
        ]

        output.append("| " + " | ".join(row) + " |")

    return "\n".join(output)


def generate_readme_condensed_table(data):
    """Generate condensed table for README.md"""
    engines = data["engines"]

    output = []
    output.append("## Quick Engine Comparison")
    output.append("")
    output.append("| Engine | Languages | Size | Key Features |")
    output.append("|--------|-----------|------|--------------|")

    # Show ALL engines
    for e in engines:

        # Get first 6-8 language flags
        flags = []
        count = 0
        for lang_data in e["languages"].values():
            if lang_data["supported"]:
                flags.append(lang_data["flag"])
                count += 1
                if count >= 6:  # Limit to 6 flags for readability
                    break

        lang_display = "".join(flags)

        # Special handling for ChatterBox 23L
        if e["id"] == "chatterbox-23l":
            lang_display = "🌐 24 languages"
        # Special handling for RVC
        elif e["id"] == "rvc":
            lang_display = "🌐 Any"
        else:
            # Add count if more languages exist
            total_langs = sum(1 for ld in e["languages"].values() if ld["supported"])
            if total_langs > 6:
                lang_display += f" +{total_langs - 6}"

        # Get 1-2 key features
        special_features = e.get("special_features", [])
        if len(special_features) > 2:
            key_features = ", ".join(special_features[:2])
        else:
            key_features = ", ".join(special_features)

        row = [
            f"**{e['name']}**",
            lang_display,
            e["size"],
            key_features
        ]

        output.append("| " + " | ".join(row) + " |")

    # Add footer with links
    output.append("")
    output.append("📊 **[Full comparison tables →](docs/ENGINE_COMPARISON.md)** | "
                  "**[Language matrix →](docs/LANGUAGE_SUPPORT.md)** | "
                  "**[Feature matrix →](docs/FEATURE_COMPARISON.md)**")

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


def inject_into_readme(condensed_table):
    """Inject condensed table into README.md between markers. Returns: True=written, False=unchanged, None=error"""
    readme_path = Path(__file__).parent.parent / "README.md"

    # Read current README
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Define markers
    start_marker = "<!-- ENGINE_COMPARISON_START -->"
    end_marker = "<!-- ENGINE_COMPARISON_END -->"

    # Check if markers exist
    if start_marker not in content or end_marker not in content:
        print("⚠️  Markers not found in README.md")
        print(f"   Please add these markers where you want the table:")
        print(f"   {start_marker}")
        print(f"   {end_marker}")
        return None  # Error

    # Replace content between markers
    pattern = f"{re.escape(start_marker)}.*?{re.escape(end_marker)}"
    replacement = f"{start_marker}\n\n{condensed_table}\n\n{end_marker}"

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Check if content actually changed
    if new_content == content:
        return False  # No change

    # Write back only if changed
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return True  # Written


def write_if_changed(file_path, new_content):
    """Write file only if content changed. Returns True if written."""
    # Check if file exists and compare content
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            existing_content = f.read()

        if existing_content == new_content:
            return False  # No change needed

    # Write the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True  # File was written


def main():
    """Generate all tables and optionally inject into README"""
    import sys

    data = load_data()
    docs_dir = Path(__file__).parent.parent / "docs"

    # Generate Engine Comparison
    print("Generating Engine Comparison table...")
    engine_comp = generate_engine_comparison(data)
    if write_if_changed(docs_dir / "ENGINE_COMPARISON.md", engine_comp):
        print(f"✅ Written to {docs_dir / 'ENGINE_COMPARISON.md'}")
    else:
        print(f"⏭️  Skipped {docs_dir / 'ENGINE_COMPARISON.md'} (unchanged)")

    # Generate Language Support
    print("Generating Language Support table...")
    lang_support = generate_language_support(data)
    if write_if_changed(docs_dir / "LANGUAGE_SUPPORT.md", lang_support):
        print(f"✅ Written to {docs_dir / 'LANGUAGE_SUPPORT.md'}")
    else:
        print(f"⏭️  Skipped {docs_dir / 'LANGUAGE_SUPPORT.md'} (unchanged)")

    # Generate Feature Comparison
    print("Generating Feature Comparison table...")
    feat_comp = generate_feature_comparison(data)
    if write_if_changed(docs_dir / "FEATURE_COMPARISON.md", feat_comp):
        print(f"✅ Written to {docs_dir / 'FEATURE_COMPARISON.md'}")
    else:
        print(f"⏭️  Skipped {docs_dir / 'FEATURE_COMPARISON.md'} (unchanged)")

    # Generate and inject condensed README table
    print("\nGenerating condensed README table...")
    condensed = generate_readme_condensed_table(data)

    # Check if --readme flag is passed
    if "--readme" in sys.argv:
        print("Injecting into README.md...")
        result = inject_into_readme(condensed)
        if result is True:
            print("✅ README.md updated successfully!")
        elif result is False:
            print("⏭️  Skipped README.md (unchanged)")
        else:  # None = error
            print("❌ README.md injection failed (markers not found)")
    else:
        print("ℹ️  Condensed table generated (use --readme flag to inject into README.md)")
        print("\nPreview:")
        print(condensed)

    print("\n✅ All tables generated successfully!")


if __name__ == "__main__":
    main()
