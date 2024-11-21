#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <cuda.h>
#include <chrono>
#include <cuda_runtime.h>

#define INF std::numeric_limits<float>::max()
#define BLOCK_SIZE 1024 

// Gerar grafo (Lista de adjacência)
void generateGraph(int numVertices, int numEdges, int *vertices, int *edges, float *weights) {
    srand(42);  // Semente fixa
    vertices[0] = 0;

    int edgeIndex = 0;

    for (int i = 0; i < numVertices - 1; i++) {
        edges[edgeIndex] = i + 1;
        weights[edgeIndex] = (float)(rand() % 10 + 1);
        edgeIndex++;
    }

    for (int i = edgeIndex; i < numEdges; i++) {
        int from = rand() % numVertices;
        int to = rand() % numVertices;

        edges[i] = to;
        weights[i] = (float)(rand() % 10 + 1);
    }

    for (int i = 0; i <= numVertices; i++) {
        vertices[i] = (i < edgeIndex) ? i : edgeIndex;
    }
}

// Sequencial no CPU
void sssp_sequential(int numVertices, int *vertices, int *edges, float *weights) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> cost(numVertices, INF);
    cost[0] = 0;

    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> pq;
    pq.push({0, 0});

    while (!pq.empty()) {
        float currCost = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        for (int i = vertices[u]; i < vertices[u + 1]; i++) {
            int v = edges[i];
            float weight = weights[i];

            if (cost[u] + weight < cost[v]) {
                cost[v] = cost[u] + weight;
                pq.push({cost[v], v});
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    std::cout << "Tempo Sequencial: " << duration.count() << " ms" << std::endl;
}

// Paralelo no GPU
__device__ float atomicMinFloat(float *address, float value) {
    int *addressAsInt = (int *)address;
    int old = *addressAsInt, assumed;

    do {
        assumed = old;
        old = atomicCAS(addressAsInt, assumed,
                        __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void relax_U(float *c, int *edges, float *weights, int *vertices, bool *u, int numVertices) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < numVertices && u[id]) {
        for (int j = vertices[id]; j < vertices[id + 1]; j++) {
            int neighbor = edges[j];
            float newCost = c[id] + weights[j];
            if (newCost < c[neighbor]) {
                atomicMinFloat(&c[neighbor], newCost);
            }
        }
    }
}

void sssp_parallel(int numVertices, int numEdges, int *vertices, int *edges, float *weights) {
    float *h_cost = (float *)malloc(numVertices * sizeof(float));
    bool *h_unresolved = (bool *)malloc(numVertices * sizeof(bool));

    for (int i = 0; i < numVertices; i++) {
        h_cost[i] = (i == 0) ? 0 : INF;
        h_unresolved[i] = true;
    }

    int *d_vertices, *d_edges;
    float *d_weights, *d_cost;
    bool *d_unresolved;

    cudaMalloc(&d_vertices, (numVertices + 1) * sizeof(int));
    cudaMalloc(&d_edges, numEdges * sizeof(int));
    cudaMalloc(&d_weights, numEdges * sizeof(float));
    cudaMalloc(&d_cost, numVertices * sizeof(float));
    cudaMalloc(&d_unresolved, numVertices * sizeof(bool));

    cudaMemcpy(d_vertices, vertices, (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges, numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, numEdges * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, h_cost, numVertices * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_unresolved, h_unresolved, numVertices * sizeof(bool), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((numVertices + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < numVertices; i++) {
        relax_U<<<grid, block>>>(d_cost, d_edges, d_weights, d_vertices, d_unresolved, numVertices);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Tempo Paralelo: " << milliseconds << " ms" << std::endl;

    free(h_cost);
    free(h_unresolved);
    cudaFree(d_vertices);
    cudaFree(d_edges);
    cudaFree(d_weights);
    cudaFree(d_cost);
    cudaFree(d_unresolved);
}

int main() {
    int numVertices = 500000;
    int numEdges = 2000000;

    int *vertices = new int[numVertices + 1];
    int *edges = new int[numEdges];
    float *weights = new float[numEdges];

    generateGraph(numVertices, numEdges, vertices, edges, weights);

    sssp_sequential(numVertices, vertices, edges, weights);
    sssp_parallel(numVertices, numEdges, vertices, edges, weights);

    delete[] vertices;
    delete[] edges;
    delete[] weights;

    return 0;
}
