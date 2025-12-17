import torch
import os
import argparse

# Import both model classes and their respective configs (aliasing custom configs)
from model import Llama3Model, LLAMA32_CONFIG_1B, LLAMA32_CONFIG_3B
from model_customrope import Llama3CustomRoPEModel, LLAMA32_CONFIG_1B as LLAMA32_CUSTOM_CONFIG_1B, LLAMA32_CONFIG_3B as LLAMA32_CUSTOM_CONFIG_3B


def main():
    parser = argparse.ArgumentParser(description="Export a Llama3.2 model to ONNX format.")
    parser.add_argument(
        "-i",
        "--input-file",
        dest="input_file",
        type=str,
        required=True,
        help="Path to the input PyTorch model file (e.g., llama3.2-1B-base.pth)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_file",
        type=str,
        required=True,
        help="Path for the output ONNX model file (e.g., llama3.2-1B-base.onnx)",
    )
    parser.add_argument(
        "-s",
        "--model-source",
        dest="model_source",
        type=str,
        default="default",
        choices=["default", "custom-rope"],
        help="Choose model source: 'default' for model.py or 'custom-rope' for model_customrope.py.",
    )
    args = parser.parse_args()

    # Assign the correct ModelClass and configs based on the --model-source argument
    if args.model_source == "custom-rope":
        print("Using model from model_customrope.py")
        ActiveLlama3Model = Llama3CustomRoPEModel
    else:
        print("Using model from model.py")
        ActiveLlama3Model = Llama3Model

    # Determine model variant from input file name
    input_filename = os.path.basename(args.input_file)
    if "1B-base" in input_filename:
        config = LLAMA32_CONFIG_1B
    elif "1B-instruct" in input_filename:
        config = LLAMA32_CONFIG_1B
    elif "3B-base" in input_filename:
        config = LLAMA32_CONFIG_3B
    elif "3B-instruct" in input_filename:
        config = LLAMA32_CONFIG_3B
    else:
        raise ValueError(
            f"Could not determine model variant from input file name: {input_filename}. "
            "Please ensure the filename contains '1B-base', '1B-instruct', '3B-base', or '3B-instruct'."
        )

    weight_filename = args.input_file

    # Modify config for ONNX export: use float32 instead of bfloat16 for broader compatibility
    export_config = config.copy()
    export_config["dtype"] = torch.float32

    # Instantiate the model with the export-friendly configuration
    print("Instantiating model for export...")
    model = ActiveLlama3Model(export_config)
    model.eval()

    # Load the pre-trained weights, converting them from bfloat16 to float32 on the fly
    print(f"Loading and converting weights from {weight_filename}...")
    state_dict_bf16 = torch.load(weight_filename, map_location="cpu")
    state_dict_fp32 = {k: v.to(torch.float32) for k, v in state_dict_bf16.items()}

    # Load the converted weights into the model
    # `strict=False` is used because buffers like 'cos' and 'sin' are not in the state_dict
    unmatched_keys = model.load_state_dict(state_dict_fp32, strict=False)
    if unmatched_keys.missing_keys:
        print(f"Info: {len(unmatched_keys.missing_keys)} keys were missing in the state dict (expected for buffers).")
    if unmatched_keys.unexpected_keys:
        print(f"Warning: {len(unmatched_keys.unexpected_keys)} keys in the state dict were not used by the model.")

    # Create a dummy input tensor for tracing the model graph
    batch_size = 1
    seq_length = 128  # Example sequence length for tracing
    dummy_input = torch.randint(0, export_config["vocab_size"], (batch_size, seq_length), dtype=torch.long)

    # Define the output path for the ONNX model
    onnx_path = args.output_file
    print(f"Exporting model to {onnx_path}...")

    # Export the model to ONNX format
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            do_constant_folding=True,
            opset_version=14,  # A reasonably modern ONNX opset version
            verbose=False,
        )
        print(f"Model exported successfully to {onnx_path}")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")


if __name__ == "__main__":
    main()
