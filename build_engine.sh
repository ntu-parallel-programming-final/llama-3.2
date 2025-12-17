#!/bin/bash

# Default values
onnx_file="./onnx/llama3.2-1B-base.onnx"
engine_file="llama3.2-1B-base.trt"
min_shapes="input_ids:1x1"
opt_shapes="input_ids:1x512"
max_shapes="input_ids:1x2048"
PLUGIN=""
other_args=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o | --onnx)
      onnx_file="$2"
      shift 2
      ;;
    -e | --saveEngine)
      engine_file="$2"
      shift 2
      ;;
    --minShapes)
      min_shapes="$2"
      shift 2
      ;;
    --optShapes)
      opt_shapes="$2"
      shift 2
      ;;
    --maxShapes)
      max_shapes="$2"
      shift 2
      ;;
    -p | --staticPlugins)
      PLUGIN="--staticPlugins=$2"
      shift 2
      ;;
    *)
      other_args="$other_args $1"
      shift
      ;;
  esac
done

trtexec --onnx="$onnx_file" \
        --saveEngine="$engine_file" \
        --minShapes="$min_shapes" \
        --optShapes="$opt_shapes" \
        --maxShapes="$max_shapes" \
        --profilingVerbosity=detailed \
        --dumpProfile \
        ${PLUGIN} \
        $other_args
