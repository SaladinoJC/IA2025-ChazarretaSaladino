import pygad
import numpy as np

# ===========================
# Función objetivo (fitness)
# ===========================
def fitness_func(ga_instance, solution, solution_idx):
    glucosa, nitrogeno, temperatura = solution
    
    # Función simulada de producción del metabolito
    produccion = (100 * 
                  np.exp(-((glucosa-25)**2)/100) *
                  np.exp(-((nitrogeno-5)**2)/4) *
                  np.exp(-((temperatura-37)**2)/2))
    return produccion

# ===========================
# Parámetros del algoritmo genético
# ===========================
num_generations = 5000
num_parents_mating = 4
sol_per_pop = 10
num_genes = 3

gene_space = [
    {'low': 0, 'high': 50},  # glucosa
    {'low': 0, 'high': 10},  # nitrogeno
    {'low': 30, 'high': 40}  # temperatura
]

# ===========================
# Crear el objeto GA
# ===========================
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       mutation_percent_genes=20,
                       mutation_type="random")

# ===========================
# Ejecutar GA
# ===========================
ga_instance.run()

# ===========================
# Obtener la mejor solución
# ===========================
solution, solution_fitness, solution_idx = ga_instance.best_solution()

# ===========================
# Predicción basada en la mejor solución
# ===========================
predicted_output = fitness_func(ga_instance, solution, solution_idx)

# ===========================
# Número de generación donde se alcanzó el mejor fitness
# ===========================
best_generation = np.argmax(ga_instance.best_solutions_fitness)

# ===========================
# Imprimir resultados
# ===========================
print("\nParameters of the best solution :")
print(solution)
print("Fitness value of the best solution =", solution_fitness)
print("Index of the best solution :", solution_idx)
print("Predicted output based on the best solution :", predicted_output)
print(f"Best fitness value reached after {best_generation+1} generations")

# ===========================
# Graficar evolución
# ===========================
ga_instance.plot_fitness()
