"""Named Step Audio EditX checkpoints, independent of engine imports."""

DEFAULT_MODEL_NAME = "Step-Audio-EditX-2026-01-23"
LEGACY_MODEL_NAME = "Step-Audio-EditX"

# Dates identify weight releases; revisions also include the compatible config.
MODEL_CHECKPOINTS = {
    DEFAULT_MODEL_NAME: {
        "revision": "5fe2f8a05c2353301ad47d3c1747b262115da138",
        "description": "Step Audio EditX - 2026-01-23 checkpoint with expanded sound tags (7GB)",
    },
    "Step-Audio-EditX-2025-11-28": {
        "revision": "7f3de603ae46c96dff6f06f47b1d5a45aabd34fe",
        "description": "Step Audio EditX - 2025-11-28 legacy checkpoint (7GB)",
    },
}

TOKENIZER_REVISION = "af7e5a3ec06175a7facae9d4100073d6e4dbb36c"
