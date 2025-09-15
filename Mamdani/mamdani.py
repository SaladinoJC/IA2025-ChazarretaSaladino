import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# ========================
# Universos de variables
# ========================
x_ext = np.arange(-10, 36, 1)   # Temperatura exterior
x_int = np.arange(40, 101, 1)   # Temperatura interior
x_flame = np.arange(0, 101, 1)  # Tamaño de la llama (%)

# ========================
# Funciones de pertenencia
# ========================
# Temperatura exterior
ext_low = fuzz.trimf(x_ext, [-10, -10, 15])
ext_med = fuzz.trimf(x_ext, [0, 15, 30])
ext_high = fuzz.trimf(x_ext, [15, 35, 35])

# Temperatura interior
int_high = fuzz.trimf(x_int, [60, 70, 80])
int_critical = fuzz.trimf(x_int, [75, 90, 100])

# Tamaño de la llama
flame_pilot = fuzz.trimf(x_flame, [0, 10, 20])
flame_mod = fuzz.trimf(x_flame, [30, 50, 70])
flame_high = fuzz.trimf(x_flame, [80, 100, 100])

# ========================
# Graficar funciones de pertenencia de entrada
# ========================
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(x_ext, ext_low, 'b', linewidth=1.5, label='Exterior baja')
ax0.plot(x_ext, ext_med, 'g', linewidth=1.5, label='Exterior media')
ax0.plot(x_ext, ext_high, 'r', linewidth=1.5, label='Exterior alta')
ax0.set_title('Temperatura exterior')
ax0.legend()
plt.tight_layout()

fig, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(x_int, int_high, 'g', linewidth=1.5, label='Interior alta')
ax1.plot(x_int, int_critical, 'r', linewidth=1.5, label='Interior crítica')
ax1.set_title('Temperatura interior')
ax1.legend()
plt.tight_layout()

# ========================
# Ejemplo de entradas
# ========================
ext_val = 30
int_val = 75

# ========================
# Grados de pertenencia
# ========================
ext_level_low = fuzz.interp_membership(x_ext, ext_low, ext_val)
ext_level_med = fuzz.interp_membership(x_ext, ext_med, ext_val)
ext_level_high = fuzz.interp_membership(x_ext, ext_high, ext_val)

int_level_high = fuzz.interp_membership(x_int, int_high, int_val)
int_level_critical = fuzz.interp_membership(x_int, int_critical, int_val)

# ========================
# Reglas difusas
# ========================
rule1 = ext_level_low          # Exterior baja → llama alta
rule2 = ext_level_med          # Exterior media → llama moderada
rule3 = ext_level_high         # Exterior alta → piloto

# ========================
# Activación de las reglas
# ========================
flame_activation_high = np.fmin(rule1, flame_high)
flame_activation_mod = np.fmin(rule2, flame_mod)
flame_activation_pilot = np.fmin(rule3, flame_pilot)

# Limitar llama por temperatura interior
flame_activation_high = np.fmin(flame_activation_high, 100 - int_level_high*30)
flame_activation_mod = np.fmin(flame_activation_mod, 100 - int_level_high*20)
flame_activation_pilot = np.fmin(flame_activation_pilot, 100 - int_level_critical*50)

# ========================
# Graficar funciones de pertenencia de salida
# ========================
flame0 = np.zeros_like(x_flame)
fig, ax2 = plt.subplots(figsize=(8,3))
ax2.fill_between(x_flame, flame0, flame_activation_high, facecolor='r', alpha=0.7)
ax2.plot(x_flame, flame_high, 'r', linestyle='--', linewidth=0.5)
ax2.fill_between(x_flame, flame0, flame_activation_mod, facecolor='g', alpha=0.7)
ax2.plot(x_flame, flame_mod, 'g', linestyle='--', linewidth=0.5)
ax2.fill_between(x_flame, flame0, flame_activation_pilot, facecolor='b', alpha=0.7)
ax2.plot(x_flame, flame_pilot, 'b', linestyle='--', linewidth=0.5)
ax2.set_title('Activaciones de salida por reglas difusas')
plt.tight_layout()

# ========================
# Agregación
# ========================
aggregated = np.fmax(flame_activation_pilot,
                     np.fmax(flame_activation_mod, flame_activation_high))

# ========================
# Defuzzificación
# ========================
flame_out = fuzz.defuzz(x_flame, aggregated, 'centroid')
flame_activation = fuzz.interp_membership(x_flame, aggregated, flame_out)
print(f"Tamaño de la llama recomendado: {flame_out:.1f}%")

# ========================
# Graficar resultado final
# ========================
fig, ax3 = plt.subplots(figsize=(8,3))
ax3.plot(x_flame, flame_high, 'r', linestyle='--', linewidth=0.5)
ax3.plot(x_flame, flame_mod, 'g', linestyle='--', linewidth=0.5)
ax3.plot(x_flame, flame_pilot, 'b', linestyle='--', linewidth=0.5)
ax3.fill_between(x_flame, flame0, aggregated, facecolor='Orange', alpha=0.7)
ax3.plot([flame_out, flame_out], [0, flame_activation], 'k', linewidth=1.5)
ax3.set_title('Salida agregada y valor defuzzificado')
plt.tight_layout()

# ========================
# Defuzzificación con varios métodos
# ========================
mfx = aggregated
defuzz_centroid = fuzz.defuzz(x_flame, mfx, 'centroid')
defuzz_bisector = fuzz.defuzz(x_flame, mfx, 'bisector')
defuzz_mom = fuzz.defuzz(x_flame, mfx, 'mom')
defuzz_som = fuzz.defuzz(x_flame, mfx, 'som')
defuzz_lom = fuzz.defuzz(x_flame, mfx, 'lom')

labels = ['centroid', 'bisector', 'mom', 'som', 'lom']
xvals = [defuzz_centroid, defuzz_bisector, defuzz_mom, defuzz_som, defuzz_lom]
colors = ['r','b','g','c','m']
ymax = [fuzz.interp_membership(x_flame, mfx, i) for i in xvals]

plt.figure(figsize=(8,5))
plt.plot(x_flame, mfx, 'k', label='Agregada')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)
plt.ylabel('Pertenencia difusa')
plt.xlabel('Tamaño de la llama (%)')
plt.title('Defuzzificación por distintos métodos')
plt.legend()
plt.show()
