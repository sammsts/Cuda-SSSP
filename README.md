# CUDA SSSP: Parallel and Sequential Implementation of the Minimum Path Algorithm

This project implements the Single Source Shortest Path (SSSP) algorithm, using CUDA for parallel execution on GPUs and a sequential version for comparison. The goal is to evaluate the performance of minimum path calculation on different graph sizes and analyze the efficiency of parallel execution on modern GPUs.

---

## **Description**

The project implements SSSP using the edge relaxation strategy. Tests were performed with randomly generated graphs to measure execution performance in two configurations:
1. **Sequential**: Using the CPU.
2. **Parallel**: Using CUDA for GPU execution (tested on NVIDIA GTX 1050).

Although the theory indicates that parallel execution should be more efficient in large and dense graphs, the results showed significant differences, influenced by hardware configuration and implementation limitations.

---

## **Project Features

- **Random Connected Graph Generation**:
  - The 'generateGraph' function generates connected graphs, configuring the edges, vertices and weights in a pseudo-random way, ensuring minimal connectivity.

- **Sequential Execution**:
  - CPU implementation using the priority queue to process edges.

- **Parallel Execution**:
  - Implementation with CUDA to process multiple vertices and edges simultaneously, using atomicMinFloat for synchronization between threads.

- **Performance Comparison**:
  - Execution times of both approaches are captured and compared graphically.

---

## **Results**

Tests indicated that the parallel version had lower performance compared to the sequential version for the hardware used (NVIDIA GTX 1050). This was due to factors such as:
- Communication overhead between CPU and GPU.
- Limited optimizations in the CUDA kernel.
- Hardware configuration.

Comparative graphs of the performance of both approaches are available in the project.

---

## **Requirements**

### **Hardware**
- CUDA-compatible GPU (e.g., NVIDIA GTX 1050 or higher).

### **Software**
- NVIDIA CUDA Toolkit.
- nvcc compiler.
- CUDA-compatible operating system (e.g., Windows or Linux).

---

## **How to Run**

1. **Clone the Repository**:
   ```bash
    git clone https://github.com/seu-usuario/cuda-sssp.git
    CUPA CUDA-SSSP
2. **Compile the Code**:
   ```bash
    nvcc -Xcompiler "/wd4244" sssp.cu -o sssp
3. **Run**:
   ```bash
    ./sssp.exe
