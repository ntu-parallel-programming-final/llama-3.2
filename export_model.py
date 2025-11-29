import torch
from model import Llama3Model, LLAMA32_CONFIG_1B, LLAMA32_CONFIG_3B
import os

def main():
    # --- Configuration ---
    # Change these variables to export a different model
    model_variant = "1B-base"  # Choices: "1B-base", "1B-instruct", "3B-base", "3B-instruct"
    # --- End Configuration ---

    if "1B" in model_variant:
        config = LLAMA32_CONFIG_1B
        weight_filename = f"llama3.2-{model_variant}.pth"
    elif "3B" in model_variant:
        config = LLAMA32_CONFIG_3B
        weight_filename = f"llama3.2-{model_variant}.pth"
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

    # Modify config for ONNX export: use float32 instead of bfloat16 for broader compatibility
    export_config = config.copy()
    export_config["dtype"] = torch.float32

    # Instantiate the model with the export-friendly configuration
    print("Instantiating model for export...")
    model = Llama3Model(export_config)
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
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"llama3.2-{model_variant}.onnx")
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
