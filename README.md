### Hybridflow

Hybridflow is a system for **automatic device selection and execution** of computational kernels written in C, C++, or Python. Our goal is to analyze source code, predict whether execution should occur on a CPU or GPU through customized ML model, and run the code on the optimal device without manual intervention.

## Overview

Hybridflow works by:
1. **Static Analysis:** Parses the input source file to extract computational features such as operation type (e.g., matrix multiplication, convolution), total size.
2. **Dynamic System State Collection:** Gathers system-level features such as device profiles and current resource availability.
3. **Feature Vector Construction:** Combines static code analysis and dynamic system features into a unified feature vector.
4. **Model Prediction:** Uses a machine learning model to predict the best device ("cpu" or "gpu") for execution.
5. **Device Execution:** Compiles and runs the code on the predicted device, generating wrappers automatically if needed.

## Features

- **Supports C, C++, and Python kernels:** Static analysis adapts to each language.
- **Automatic wrapper generation:** If the code lacks a main function, Hybridflow generates a wrapper for standalone execution.
- **Advanced code analysis:** Uses AST parsing to extract operation types, memory requirements, loop bounds, reduction patterns, and more.
- **Device-aware execution:** Can compile with GCC for CPU, NVCC for GPU (CUDA), or run Python scripts directly.
- **Extensible collector modules:** Supports collection and benchmarking for common operations (matmul, conv2d, elementwise, reduce, transpose, sort, scan, fft).

## Usage

```bash
python main.py <source_file>
```

- `<source_file>` can be `.c`, `.cpp`, or `.py`.

Hybridflow will:
- Analyze the file
- Predict the optimal device based on extracted features and system state
- Compile and execute the code on that device
- Log results and errors

## Architecture

- **main.py**: Pipeline entry; coordinates analysis, prediction, and execution.
- **analyzer/ast_parser.py**: Core AST parser for C/C++ and Python, extracts features for prediction.
- **collector/device_runner.py**: Handles compilation and execution logic for CPU/GPU.
- **model/preprocessing_model.pkl**: trained model
- **utils/logger.py**: Logging utilities

## Extending

- Run the same on edge device also like Jetson Nano
- Create a library support, simple import and play functionality
- Add the support for automated CUDA code generation
- Try to integrate this with the compiler to make it full automated

## Example Operations Detected

- Matrix Multiplication (`matmul`)
- Convolution (`conv2d`)
- Elementwise operations
- Reduction
- Transpose
- Sort
- Scan
- FFT

## Requirements

- Python 3.x
- GCC for CPU compilation
- NVCC (CUDA) for GPU kernels
- Clang Python bindings (`clang.cindex`)
- Other dependencies as required by submodules

## Authors

- Bala Vignesh 
- Harshvardhan Parikh
- Nikunj Bhatt

---
