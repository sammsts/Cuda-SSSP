import matplotlib.pyplot as plt

# Dados de exemplo
vertices = [3, 6, 10, 50, 1000]
tempos = [0.10, 0.25, 0.75, 3.40, 125.60]

plt.figure(figsize=(8, 6))
plt.plot(vertices, tempos, marker='o', linestyle='-', label='Desempenho CUDA')
plt.title("Desempenho do Algoritmo SSSP")
plt.xlabel("Número de Vértices")
plt.ylabel("Tempo de Execução (ms)")
plt.grid()
plt.legend()
plt.savefig("desempenho_sssp.png")
plt.show()
