import matplotlib.pyplot as plt

# Dados coletados
vertices = [1000, 2000, 5000, 50000]
tempos_cuda = [54.61, 95.98, 246.06, 2506.67]  # Exemplos
tempos_sequencial = [0.1442, 0.2926, 0.7185, 7.3478]  # Exemplos

# Criando o gráfico
plt.figure(figsize=(10, 6))
plt.plot(vertices, tempos_cuda, marker='o', linestyle='-', label='CUDA')
plt.plot(vertices, tempos_sequencial, marker='s', linestyle='--', label='Sequencial')
plt.title("Comparação de Desempenho: CUDA vs Sequencial")
plt.xlabel("Número de Vértices")
plt.ylabel("Tempo de Execução (ms)")
plt.legend()
plt.grid()
plt.savefig("comparacao_desempenho.png")
plt.show()
