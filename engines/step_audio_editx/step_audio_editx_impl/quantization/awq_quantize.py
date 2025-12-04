#!/usr/bin/env python3
"""
AWQ Quantization Script
Performs AWQ quantization processing using llmcompressor

Usage:
    python awq_quantize.py --model_path /path/to/model [other options]

Dependencies: llmcompressor
"""

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Optional, Dict
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor import oneshot
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from transformers import AutoModelForCausalLM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from datasets import load_dataset

# Set longer timeout for dataset downloads
# os.environ.setdefault('HF_DATASETS_TIMEOUT', '120')
# os.environ.setdefault('REQUESTS_TIMEOUT', '120')
os.environ['TORCH_FX_DISABLE'] = '1'  # Disable torch.fx completely
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting


def validate_model_path(model_path: str) -> Path:
    """Validate if the model path is valid"""
    path = Path(model_path)

    if not path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Check if it contains necessary model files
    expected_files = ['config.json']
    model_files = ['pytorch_model.bin', 'model.safetensors']

    has_config = any((path / f).exists() for f in expected_files)
    has_model = any((path / f).exists() for f in model_files) or any(
        list(path.glob('pytorch_model-*.bin')) or list(path.glob('model-*.safetensors'))
    )

    if not (has_config and has_model):
        print(f"Model path may be incomplete. Detected files: {list(path.iterdir())[:10]}")

    return path


def create_output_directory(model_path: Path, suffix: str = "awq-4bit") -> Path:
    """Create output subdirectory under the model path"""
    output_dir = model_path / suffix
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


def get_awq_recipe(
    scheme: str,
    group_size: int = 128,
    ignore_layers: Optional[list] = None
) -> list:
    """
    Create AWQ quantization configuration

    Args:
        scheme: Quantization scheme, default W4A16_ASYM (4-bit weights, 16-bit activations, asymmetric)
        group_size: Group size, default 128
        ignore_layers: List of layers to ignore

    Returns:
        AWQ configuration list
    """
    if not ignore_layers:
        ignore_layers = [
            "lm_head",
            "embed_tokens",
            "model.embed_tokens", 
            "model.norm",
            "norm",
            "output",
            "classifier"
        ]
    if not scheme:
        # Create quantization arguments using compressed_tensors objects
        weights_quant_args = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            symmetric=True,
            strategy=QuantizationStrategy.GROUP,
            group_size=group_size,
            observer="minmax"
        )

        awq_modifier = AWQModifier(
            offload_device=torch.device("cpu"),
            scheme=None,
            ignore=ignore_layers,
            # targets=["Linear"],
            config_groups = {
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=weights_quant_args,
                    input_activations=None,
                )
            }
        )
    else:
        awq_modifier = AWQModifier(
            scheme=scheme,       
            ignore=ignore_layers,
            targets=["Linear"]   
        )
    recipe = [
        # SmoothQuantModifier(smoothing_strength=0.5), 
        awq_modifier,
    ]
    return recipe


def quantize_model(
    model_path: str,
    output_dir: str,
    scheme: str,
    dataset: str = "open_platypus",
    max_seq_length: int = 2048,
    num_calibration_samples: int = 512,
    group_size: int = 128,
    ignore_layers: Optional[list] = None,
    device: Optional[str] = None
) -> None:
    """
    Perform AWQ quantization

    Args:
        model_path: Original model path
        output_dir: Output directory path
        dataset: Calibration dataset name or local path
        max_seq_length: Maximum sequence length
        num_calibration_samples: Number of calibration samples
        scheme: Quantization scheme
        group_size: Group size
        ignore_layers: List of layers to ignore
        device: Specified device
    """

    print(f"Preparing calibration dataset: {dataset}")
    # Try to load dataset with datasets library
    try:
        print(f"Checking if '{dataset}' contains '/': {'/' in dataset}")
        if '/' in dataset:  # Likely a path or repo name
            print(f"Attempting to load dataset from: {dataset}")
            # Check if it's a local file
            if dataset.endswith('.json'):
                dataset = load_dataset('json', data_files=dataset, split='train')
            elif dataset.endswith('.csv'):
                dataset = load_dataset('csv', data_files=dataset, split='train')
            elif dataset.endswith('.parquet'):
                dataset = load_dataset('parquet', data_files=dataset, split='train')
            else:
                # Assume it's a directory or HuggingFace dataset name
                dataset = load_dataset(dataset, split='train')
            print(f"Loaded dataset with {len(dataset)} samples")

            # Add text column if missing (required by llmcompressor)
            if 'text' not in dataset.column_names:
                print(f"Dataset columns: {dataset.column_names}")
                print("Adding 'text' column...")

                def create_text(examples):
                    texts = []
                    print(f'examples.keys(): {list(examples.keys())}')

                    # Get the three columns: system, human, assistant
                    system_msgs = examples.get('0', [])  # Column 0: System messages
                    human_msgs = examples.get('1', [])   # Column 1: Human messages
                    assistant_msgs = examples.get('2', [])  # Column 2: Assistant messages

                    print(f'system_msgs length: {len(system_msgs)}')
                    print(f'human_msgs length: {len(human_msgs)}')
                    print(f'assistant_msgs length: {len(assistant_msgs)}')

                    # Ensure all columns have the same length
                    min_length = min(len(system_msgs), len(human_msgs), len(assistant_msgs))
                    print(f'Processing {min_length} conversations')

                    for i in range(min_length):
                        conversation_parts = {
                            'system': '',
                            'human': '',
                            'assistant': ''
                        }

                        # Extract system content
                        if isinstance(system_msgs[i], dict):
                            conversation_parts['system'] = system_msgs[i].get('value', '')

                        # Extract human content
                        if isinstance(human_msgs[i], dict):
                            conversation_parts['human'] = human_msgs[i].get('value', '')

                        # Extract assistant content
                        if isinstance(assistant_msgs[i], dict):
                            content = assistant_msgs[i].get('value', '')
                            if isinstance(content, list):
                                # Audio tokens - keep as comma-separated string
                                conversation_parts['assistant'] = '[' + ', '.join(map(str, content)) + ']'
                            else:
                                conversation_parts['assistant'] = content

                        # Format as system\n{content}\nhuman\n{content}\nassistant\n{content}
                        formatted_text = f"system\n{conversation_parts['system']}\nhuman\n{conversation_parts['human']}\nassistant\n{conversation_parts['assistant']}"
                        texts.append(formatted_text)
                    print(f"Created {len(texts)} conversation texts")
                    return {'text': texts}

                dataset = dataset.map(create_text, batched=True)
                print("Added 'text' column for llmcompressor")
        else:
            print(f"Dataset '{dataset}' doesn't contain '/', treating as dataset name")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to original dataset string")
        pass  # Keep original dataset string for llmcompressor

    print("Starting AWQ quantization...")
    print(f"Model path: {model_path}")
    print(f"Output path: {output_dir}")
    print(f"Quantization scheme: {scheme}")
    print(f"Group size: {group_size}")
    print(f"Calibration dataset: {dataset}")
    print(f"Calibration samples: {num_calibration_samples}")

    # Check GPU availability
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Create AWQ configuration
    recipe = get_awq_recipe(scheme, group_size, ignore_layers)

    try:
        start_time = time.time()
        # load_kwargs = {
        #     "device_map": "auto",
        #     "trust_remote_code": True,
        #     "local_files_only": True
        # }
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     **load_kwargs
        # )
        # Perform one-shot quantization
        oneshot(
            model=model_path,
            dataset=dataset,
            recipe=recipe,
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            num_calibration_samples=num_calibration_samples,
            precision="auto",
            trust_remote_code_model=True,
            clear_sparse_session=True,
            save_compressed=True, 
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"Quantization completed! Time taken: {duration:.2f} seconds")
        print(f"Quantized model saved to: {output_dir}")

        # Check output files
        output_path = Path(output_dir)
        if output_path.exists():
            files = list(output_path.iterdir())
            print(f"Number of output files: {len(files)}")
            for file in files[:10]:  # Show first 10 files
                print(f"  - {file.name}")

    except Exception as e:
        print(f"Error occurred during quantization: {e}")
        raise


def estimate_memory_requirements(model_path: str) -> Dict[str, float]:
    """Estimate memory requirements"""
    try:
        # Simple file size estimation
        path = Path(model_path)
        total_size = 0

        for file in path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size

        size_gb = total_size / (1024**3)

        # Estimate memory needed for quantization (original model + intermediate quantization data + output model)
        estimated_memory = {
            "model_size_gb": size_gb,
            "estimated_peak_memory_gb": size_gb * 2.5,  # Empirical value
            "quantized_size_gb": size_gb * 0.3,  # 4-bit quantization reduces ~70%
        }

        return estimated_memory

    except Exception as e:
        print(f"Cannot estimate memory requirements: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Perform AWQ quantization using llmcompressor")

    # Required arguments
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        required=True,
        help="Model path (local directory or HuggingFace model name)"
    )

    # Optional arguments
    parser.add_argument(
        "--output_suffix", "-o",
        type=str,
        default="awq-4bit",
        help="Output subdirectory name suffix (default: awq-4bit)"
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="open_platypus",
        help="Calibration dataset (default: open_platypus)"
    )

    parser.add_argument(
        "--scheme", "-s",
        type=str,
        default="",
        choices=["W4A16_ASYM", "W4A16_SYM", "W8A16"],
        help="Quantization scheme (default: W4A16_ASYM)"
    )

    parser.add_argument(
        "--group_size", "-g",
        type=int,
        default=128,
        help="Quantization group size (default: 128)"
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)"
    )

    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)"
    )

    parser.add_argument(
        "--ignore_layers",
        type=str,
        nargs="+",
        default=[],
        help="List of layer names to ignore for quantization (default: lm_head)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Computing device (default: auto)"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show configuration information, do not perform quantization"
    )

    args = parser.parse_args()

    try:
        # Validate model path
        model_path = validate_model_path(args.model_path)

        # Create output directory
        output_dir = create_output_directory(model_path, args.output_suffix)

        # Estimate memory requirements
        memory_info = estimate_memory_requirements(str(model_path))
        if memory_info:
            print("Memory requirement estimation:")
            for key, value in memory_info.items():
                print(f"  {key}: {value:.2f}")

        # Device configuration
        device = args.device if args.device != "auto" else None

        # Display configuration information
        print("=== Quantization Configuration ===")
        print(f"Model path: {model_path}")
        print(f"Output directory: {output_dir}")
        print(f"Quantization scheme: {args.scheme}")
        print(f"Group size: {args.group_size}")
        print(f"Calibration dataset: {args.dataset}")
        print(f"Calibration samples: {args.num_calibration_samples}")
        print(f"Maximum sequence length: {args.max_seq_length}")
        print(f"Ignored layers: {args.ignore_layers}")
        print(f"Device: {device or 'auto'}")

        if args.dry_run:
            print("Dry run mode, exiting")
            return

        # Perform quantization
        quantize_model(
            model_path=str(model_path),
            output_dir=str(output_dir),
            scheme=args.scheme,
            dataset=args.dataset,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_calibration_samples,
            group_size=args.group_size,
            ignore_layers=args.ignore_layers,
            device=device
        )

        print("Quantization script execution completed!")

    except Exception as e:
        print(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()