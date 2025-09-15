import pygad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# Configuración del problema
# =========================
num_cities = np.random.randint(10, 21)
cities = np.random.rand(num_cities, 2) * 100

# =========================
# Funciones auxiliares
# =========================
def total_distance(order):
    route = [0] + list(order) + [0]
    distance = 0
    for i in range(len(route)-1):
        distance += np.linalg.norm(cities[route[i]] - cities[route[i+1]])
    return distance

def fitness_func(ga_instance, solution, solution_idx):
    dist = total_distance(solution)
    return 1e6 / (dist + 1e-6)

def nearest_neighbor(cities):
    n = len(cities)
    unvisited = set(range(1, n))
    route = [0]
    while unvisited:
        last = route[-1]
        next_city = min(unvisited, key=lambda i: np.linalg.norm(cities[last]-cities[i]))
        route.append(next_city)
        unvisited.remove(next_city)
    route.append(0)
    return route

def initial_population(pop_size, num_genes):
    pop = []
    nn = nearest_neighbor(cities)[1:-1]
    pop.append(np.array(nn))
    for _ in range(pop_size - 1):
        perm = np.random.permutation(num_genes) + 1
        pop.append(perm)
    return np.array(pop)

# =========================
# Crossover OX personalizado
# =========================
def ordered_crossover(parents, offspring_size, ga_instance):
    offspring = []
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        p1 = parents[parent1_idx]
        p2 = parents[parent2_idx]
        start, end = np.sort(np.random.choice(range(len(p1)), size=2, replace=False))
        child = [-1]*len(p1)
        child[start:end+1] = p1[start:end+1]
        p2_genes = [g for g in p2 if g not in child]
        ptr = 0
        for i in range(len(child)):
            if child[i] == -1:
                child[i] = p2_genes[ptr]
                ptr += 1
        offspring.append(child)
    return np.array(offspring)

# =========================
# Historia de generación
# =========================
best_routes = []
best_distances = []

def on_generation(ga_instance):
    generation = ga_instance.generations_completed
    if generation % 500 == 0:
        print(f"Generación {generation} completada")
    if generation % 100 == 0 or generation == ga_instance.num_generations:
        solution, _, _ = ga_instance.best_solution()
        best_routes.append(solution.copy())
        best_distances.append(total_distance(solution))

# =========================
# Configuración GA
# =========================
num_generations = 5000
num_parents_mating = 5
sol_per_pop = 50
num_genes = num_cities - 1

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_type=int,
    parent_selection_type="rank",
    crossover_type=ordered_crossover,  # usamos función personalizada
    mutation_type="inversion",
    mutation_percent_genes=5,
    initial_population=initial_population(sol_per_pop, num_genes),
    on_generation=on_generation
)

# =========================
# Ejecutar GA
# =========================
ga_instance.run()

best_solution, best_fitness, best_idx = ga_instance.best_solution()
best_distance = total_distance(best_solution)
best_route_full = [0] + list(map(int, best_solution)) + [0]
print("\nMejor orden de ciudades encontrado:", best_route_full)
print("Distancia total mínima:", best_distance)
print("Fitness de la mejor solución:", best_fitness)

# =========================
# Ruta vecino más cercano
# =========================
nn_route = nearest_neighbor(cities)
nn_distance = sum(np.linalg.norm(cities[nn_route[i]] - cities[nn_route[i+1]]) for i in range(len(nn_route)-1))
print("Distancia ruta vecino más cercano:", nn_distance)

# =========================
# Animación
# =========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

line, = ax1.plot([], [], 'o-', color='orange', markersize=8)
final_line, = ax1.plot([], [], 'o-', color='red', linewidth=2, markersize=8)
texts = [ax1.text(0,0,"") for _ in range(num_cities)]
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.set_title("Mejor ruta por generación")

ax2.set_xlim(0, len(best_routes))
ax2.set_ylim(min(best_distances)*0.95, max(best_distances)*1.05)
dist_line, = ax2.plot([], [], 'b-', linewidth=2)
ax2.set_title("Distancia mínima por generación")
ax2.set_xlabel("Paso (cada 100 generaciones)")
ax2.set_ylabel("Distancia")

def update(frame):
    route = [0] + list(best_routes[frame]) + [0]
    coords = np.array([cities[i] for i in route])
    line.set_data(coords[:,0], coords[:,1])
    final_coords = np.array([cities[i] for i in best_route_full])
    final_line.set_data(final_coords[:,0], final_coords[:,1])
    for idx, t in enumerate(texts):
        t.set_position((cities[idx,0]+0.5, cities[idx,1]+0.5))
        t.set_text(f"C{idx}")
    ax1.set_xlabel(f"Paso {frame+1} (Generaciones {frame*100}-{frame*100+100})")
    dist_line.set_data(range(frame+1), best_distances[:frame+1])
    return line, final_line, *texts, dist_line

ani = FuncAnimation(fig, update, frames=len(best_routes), blit=True, interval=50, repeat=False)
plt.tight_layout()
plt.show()
