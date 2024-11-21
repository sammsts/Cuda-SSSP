#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <cuda.h>

#define INF std::numeric_limits<float>::max()
#define BLOCK_SIZE 256

// Função para gerar o grafo (compartilhada entre sequencial e paralelo)
void generateGraph(int numVertices, int numEdges, int *vertices, int *edges, float *weights) {
    srand(42);  // Semente fixa para garantir mesma geração de dados
    vertices[0] = 0;

    for (int i = 0; i < numVertices; i++) {
        int numEdgesPerVertex = rand() % (numEdges / numVertices) + 1;  // Distribuição de arestas
        vertices[i + 1] = vertices[i] + numEdgesPerVertex;

        for (int j = vertices[i]; j < vertices[i + 1]; j++) {
            edges[j] = rand() % numVertices;  // Conexões aleatórias
            weights[j] = (float)(rand() % 10 + 1);  // Pesos aleatórios (1 a 10)
        }
    }
}

// Implementação do algoritmo SSSP sequencial
void sssp_sequential(int numVertices, int *vertices, int *edges, float *weights) {
    std::vector<float> cost(numVertices, INF);
    cost[0] = 0;  // Custo do vértice inicial é 0

    // Usar uma fila de prioridade para processar os vértices
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> pq;
    pq.push({0, 0});  // Custo e vértice inicial

    while (!pq.empty()) {
        float currCost = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        // Relaxe todas as arestas do vértice atual
        for (int i = vertices[u]; i < vertices[u + 1]; i++) {
            int v = edges[i];
            float weight = weights[i];

            if (cost[u] + weight < cost[v]) {
                cost[v] = cost[u] + weight;
                pq.push({cost[v], v});
            }
        }
    }

    // Exibir os custos finais
    std::cout << "Custos Finais (Sequencial):" << std::endl;
    for (int i = 0; i < numVertices; i++) {
        std::cout << "Vértice " << i << ": " << cost[i] << std::endl;
    }
}

// Implementação do SSSP paralelo (CUDA)
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
    // Alocação no host
    float *h_cost = (float *)malloc(numVertices * sizeof(float));
    bool *h_unresolved = (bool *)malloc(numVertices * sizeof(bool));

    for (int i = 0; i < numVertices; i++) {
        h_cost[i] = (i == 0) ? 0 : INF;
        h_unresolved[i] = true;
    }

    // Alocação na GPU
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

    // Executa o relaxamento das arestas
    for (int i = 0; i < numVertices; i++) {
        relax_U<<<grid, block>>>(d_cost, d_edges, d_weights, d_vertices, d_unresolved, numVertices);
        cudaDeviceSynchronize();
    }

    // Copia os resultados de volta
    cudaMemcpy(h_cost, d_cost, numVertices * sizeof(float), cudaMemcpyDeviceToHost);

    // Exibe os custos finais
    printf("Custos Finais (Paralelo):\n");
    for (int i = 0; i < numVertices; i++) {
        printf("Vértice %d: %.2f\n", i, h_cost[i]);
    }

    // Libera memória
    free(h_cost);
    free(h_unresolved);
    cudaFree(d_vertices);
    cudaFree(d_edges);
    cudaFree(d_weights);
    cudaFree(d_cost);
    cudaFree(d_unresolved);
}

int main() {
    int numVertices = 3;
    int numEdges = 3;

    // Aloca memória para o grafo
    int *vertices = new int[numVertices + 1];
    int *edges = new int[numEdges];
    float *weights = new float[numEdges];

    // Gera o grafo apenas uma vez
    generateGraph(numVertices, numEdges, vertices, edges, weights);

    // Log para verificar o grafo gerado
    std::cout << "=== Grafo Gerado ===" << std::endl;
    std::cout << "Vertices: ";
    for (int i = 0; i <= numVertices; i++) std::cout << vertices[i] << " ";
    std::cout << std::endl;

    std::cout << "Edges: ";
    for (int i = 0; i < numEdges; i++) std::cout << edges[i] << " ";
    std::cout << std::endl;

    std::cout << "Weights: ";
    for (int i = 0; i < numEdges; i++) std::cout << weights[i] << " ";
    std::cout << std::endl;

    // Executa o SSSP sequencial
    std::cout << "\n=== Executando SSSP Sequencial ===" << std::endl;
    sssp_sequential(numVertices, vertices, edges, weights);

    // Executa o SSSP paralelo
    std::cout << "\n=== Executando SSSP Paralelo ===" << std::endl;
    sssp_parallel(numVertices, numEdges, vertices, edges, weights);

    // Libera memória alocada
    delete[] vertices;
    delete[] edges;
    delete[] weights;

    return 0;
}
