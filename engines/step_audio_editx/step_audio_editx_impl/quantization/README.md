# AWQ Quantization

This module provides AWQ (Activation-aware Weight Quantization) quantization using llmcompressor to compress language models while maintaining high inference quality.

## Overview

AWQ is a post-training quantization method that reduces model size and improves inference speed by quantizing weights to lower precision while keeping activations at higher precision. This implementation supports various quantization schemes optimized for different use cases.

## Installation

Make sure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run AWQ quantization with minimal configuration:

```bash
python quantization/awq_quantize.py --model_path /path/to/your/model
```

This creates an `awq-4bit` subfolder within your model directory containing the quantized model, ready for inference.

### Advanced Usage

For more control over the quantization process:

```bash
python quantization/awq_quantize.py \
    --model_path /path/to/your/model \
    --output_suffix "awq-4bit" \
    --group_size 128 \
    --max_seq_length 4096 \
    --num_calibration_samples 512 \
    --dataset /path/to/your/calibration/dataset
```

## Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `--model_path` | Path to the model to quantize | - | Yes |
| `--output_suffix` | Name of the output subfolder | `awq-4bit` | No |
| `--scheme` | Quantization scheme (see below) | `W4A16_ASYM` | No |
| `--group_size` | Quantization group size | `128` | No |
| `--max_seq_length` | Maximum sequence length for calibration | `4096` | No |
| `--num_calibration_samples` | Number of calibration samples | `512` | No |
| `--dataset` | Custom calibration dataset path | `open_platypus` | No |
| `--ignore_layers` | Layer names to ignore for quantization | `["lm_head"]` | No |
| `--device` | Computing device (auto/cuda/cpu) | `auto` | No |
| `--log_level` | Log level (DEBUG/INFO/WARNING/ERROR) | `INFO` | No |
| `--dry_run` | Show config only, don't quantize | `False` | No |

## Quantization Schemes

| Scheme | Description | Use Case |
|--------|-------------|----------|
| **W4A16_ASYM** | 4-bit weights, 16-bit activations, asymmetric | **Recommended**: High compression with good quality |
| **W4A16_SYM** | 4-bit weights, 16-bit activations, symmetric | Alternative 4-bit option |
| **W8A16** | 8-bit weights, 16-bit activations | Conservative: Higher precision, larger size |

## Output

The script creates a quantized model in a subfolder within your specified model path:
```
/path/to/your/model/
├── original_model_files...
└── awq-4bit/  # or your custom suffix
    ├── quantized_model_files...
    └── quantization_config.json
```

The quantized model is immediately ready for inference and can be loaded using standard model loading procedures.

## Requirements

- Python 3.10+
- Sufficient RAM for model loading
- CUDA-compatible GPU (recommended for larger models)

## Notes

- Quantization time depends on model size and number of calibration samples
- The calibration dataset significantly impacts quantization quality
- Monitor GPU memory usage during quantization of large models
- Use `--dry_run` to preview the quantization configuration before running
- The `--ignore_layers` parameter helps preserve important layers like output heads

## Troubleshooting

**Issue**: Out of memory errors during quantization
**Solution**:
- Reduce `--num_calibration_samples` (e.g., from 512 to 256)
- Use `--device cpu` for CPU-only quantization (slower but uses less GPU memory)
- Ensure sufficient system RAM is available

**Issue**: Poor quality after quantization
**Solution**:
- Try using `W8A16` scheme for higher precision
- Increase `--num_calibration_samples` for better calibration
- Use a representative calibration dataset similar to your target use case

**Issue**: Specific layers causing issues
**Solution**: Add problematic layer names to `--ignore_layers` to exclude them from quantization