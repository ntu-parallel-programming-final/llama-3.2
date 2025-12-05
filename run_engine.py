import os
import time
import torch
import numpy as np
import tensorrt as trt
import ctypes
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
from tokenizer import Llama3Tokenizer
from model import LLAMA32_CONFIG_1B, text_to_token_ids, token_ids_to_text

# --- Configuration ---
TOKENIZER_FILE = "tokenizer.model"
PROMPT = "We shall go on to the end, we shall fight in France"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.0
TOP_K = 1
# --- End Configuration ---


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class LayerProfiler(trt.IProfiler):
    def __init__(self):
        super(LayerProfiler, self).__init__()
        self.layer_times = {}
        self.total_time = 0

    def report_layer_time(self, layer_name, ms):
        if layer_name not in self.layer_times:
            self.layer_times[layer_name] = 0
        self.layer_times[layer_name] += ms
        self.total_time += ms

    def print_layer_times(self):
        """Prints the layer times nicely."""
        total_time = sum(self.layer_times.values())
        if total_time == 0:
            print("No layer timing information collected.")
            return

        print("\n--- TensorRT Layer-wise Profile ---")
        # Sort layers by their execution time in descending order
        sorted_layers = sorted(self.layer_times.items(), key=lambda item: item[1], reverse=True)

        for layer_name, time_ms in sorted_layers:
            percentage = (time_ms / total_time) * 100
            print(f"{layer_name:80s} {time_ms:>8.4f} ms ({percentage:.2f}%)")

        print("-" * 100)
        print(f"Total Profiled Time: {total_time:.4f} ms")
        print("Note: The times are accumulated over all generation steps.")
        print("-" * 100)


def load_engine(engine_file_path):
    """
    Deserializes a TensorRT engine from a file.
    """
    print(f"Reading engine from file {engine_file_path}")
    if not os.path.exists(engine_file_path):
        raise FileNotFoundError(f"Engine file not found: {engine_file_path}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def generate_with_trt(engine, tokenizer, prompt, max_new_tokens, temperature, top_k, profiler=None):
    """
    Generates text using a TensorRT engine with an auto-regressive loop.
    """
    # --- 1. Initialization ---
    stream = cuda.Stream()
    context = engine.create_execution_context()
    if profiler:
        context.profiler = profiler

    # --- 2. Tokenize Prompt ---
    input_ids = text_to_token_ids(prompt, tokenizer)

    # --- 3. Allocate Buffers using modern TRT API ---
    input_tensor_name = "input_ids"
    output_tensor_name = "logits"

    # Validate that the tensor names exist in the engine
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    if input_tensor_name not in tensor_names or output_tensor_name not in tensor_names:
        raise ValueError(f"Engine does not have expected tensor names: '{input_tensor_name}', '{output_tensor_name}'")

    input_dtype = trt.nptype(engine.get_tensor_dtype(input_tensor_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_tensor_name))

    # Get max shapes from profile 0
    input_shape = engine.get_tensor_profile_shape(input_tensor_name, 0)[2]  # Max shape

    # Set the input shape to the max shape to determine the corresponding output shape
    context.set_input_shape(input_tensor_name, tuple(input_shape))
    output_shape = context.get_tensor_shape(output_tensor_name)

    # Allocate device memory
    d_input = cuda.mem_alloc(int(np.prod(tuple(input_shape)) * np.dtype(input_dtype).itemsize))
    d_output = cuda.mem_alloc(int(np.prod(tuple(output_shape)) * np.dtype(output_dtype).itemsize))

    # Allocate page-locked host memory for output
    h_output = cuda.pagelocked_empty(tuple(output_shape), dtype=output_dtype)

    # Set tensor addresses
    context.set_tensor_address(input_tensor_name, int(d_input))
    context.set_tensor_address(output_tensor_name, int(d_output))

    # --- 4. Generation Loop (with profiling) ---
    start_time = time.time()

    # Create CUDA events for profiling
    start_event = cuda.Event()
    end_event = cuda.Event()
    total_gpu_time_ms = 0

    for _ in range(max_new_tokens):
        current_seq_len = input_ids.shape[1]

        # a. Set the dynamic input shape for this iteration
        context.set_input_shape(input_tensor_name, (1, current_seq_len))

        # b. Prepare input data and copy to device
        input_data = np.ascontiguousarray(input_ids.numpy(), dtype=input_dtype)
        cuda.memcpy_htod_async(d_input, input_data, stream)

        # c. Run inference with profiling
        start_event.record(stream)
        context.execute_async_v3(stream.handle)
        end_event.record(stream)

        # d. Copy output from device to host
        cuda.memcpy_dtoh_async(h_output, d_output, stream)

        # e. Synchronize stream and calculate GPU time
        stream.synchronize()
        total_gpu_time_ms += end_event.time_since(start_event)


        # f. Process logits
        output_dims = context.get_tensor_shape(output_tensor_name)

        # Slice the host buffer to the actual output size
        logits_full = torch.from_numpy(h_output)
        logits = logits_full[:, :output_dims[1], :]
        
        logits = logits[:, -1, :] # Get logits for the last token

        # g. Sample the next token
        if temperature > 0.0:
            if top_k:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')), logits)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        # h. Append to the sequence
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

    end_time = time.time()

    # --- Performance Results ---
    total_wall_time_s = end_time - start_time
    avg_gpu_time_ms = total_gpu_time_ms / max_new_tokens
    print(f"\n--- Performance ---")

    print(f"Total GPU inference time: {total_gpu_time_ms:.2f} ms")
    print(f"Average GPU inference time per token: {avg_gpu_time_ms:.2f} ms")
    print(f"Tokens per second (GPU only): {1000.0 / avg_gpu_time_ms:.2f}")
    print(f"Tokens per second (end-to-end): {max_new_tokens / total_wall_time_s:.2f}")

    if profiler:
        profiler.print_layer_times()

    # --- 5. Cleanup is handled by pycuda.autoinit ---

    # --- 6. Decode and Return ---
    return token_ids_to_text(input_ids, tokenizer)

def main():
    """
    Main function to run the TensorRT generation example.
    """
    parser = argparse.ArgumentParser(description="Run inference with a Llama 3.2 TensorRT engine.")
    parser.add_argument("-e", "--engine-file", type=str, help="Path to the TensorRT engine file.")
    parser.add_argument("-p", "--prompt", type=str, default=PROMPT, help="The prompt for the model.")
    parser.add_argument("-t", "--tokenizer-file", type=str, default=TOKENIZER_FILE, help="Path to the tokenizer model file.")
    parser.add_argument("--profile", action="store_true", help="Enable layer-wise profiling.")
    args = parser.parse_args()

    profiler = None
    if args.profile:
        profiler = LayerProfiler()
        print("Layer-wise profiling enabled.")

    print("Loading tokenizer...")
    tokenizer = Llama3Tokenizer(args.tokenizer_file)

    print("Loading TensorRT engine...")
    engine = load_engine(args.engine_file)

    print("Generating text...")
    output_text = generate_with_trt(
        engine=engine,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        profiler=profiler,
    )

    print("\n--- Prompt ---")
    print(args.prompt)
    print("\n--- Generated Text ---")
    print(output_text)

if __name__ == "__main__":
    main()
