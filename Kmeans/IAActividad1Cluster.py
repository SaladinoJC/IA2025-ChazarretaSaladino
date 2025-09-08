import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects


def k_means(data, k, max_iters=100, tol=1e-4):
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]
    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        old_centroids = centroids.copy()
        for i in range(k):
            if np.any(labels == i):
                centroids[i] = data[labels == i].mean(axis=0)
        if np.all(np.linalg.norm(centroids - old_centroids, axis=1) < tol):
            break
    return centroids, labels

def graficar(data, centroids, labels):
    dim = data.shape[1]
    k = len(centroids)
    # Elegir paleta de colores complementaria
    palette = sns.color_palette("colorblind", k)
    
    if dim == 2:
        plt.figure(figsize=(6, 6))
        # Scatter de puntos
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=palette, s=60, edgecolor=None)
        # Centroides con números
        for i, (x, y) in enumerate(centroids):
            txt = plt.text(x, y, str(i), color='black', fontsize=14, fontweight='bold',ha='center', va='center')
            txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),path_effects.Normal()])
        plt.title("Clustering en 2D")
        plt.show()
    
    elif dim == 3:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        # Scatter de puntos
        for i in range(k):
            ax.scatter(data[labels == i, 0], data[labels == i, 1], data[labels == i, 2],
                       color=palette[i], s=60)
        # Centroides con números
        for i, (x, y, z) in enumerate(centroids):
            ax.text(x, y, z, str(i), color='red', fontsize=14, fontweight='bold', ha='center', va='center')
        ax.set_title("Clustering en 3D")
        plt.show()
    
    else:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroids_2d = pca.transform(centroids)
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=labels, palette=palette, s=60, edgecolor=None)
        for i, (x, y) in enumerate(centroids_2d):
            plt.text(x, y, str(i), color='red', fontsize=14, fontweight='bold', ha='center', va='center')
        plt.title(f"Clustering en {dim}D (reducido a 2D con PCA)")
        plt.show()



def main():
    tipo = '3D'    #puede ser 2D 3D 5D
    if tipo== '3D':
        np.random.seed(0)
        n_per_cluster = 30
        c1 = np.random.randn(n_per_cluster, 3) * 0.5 + np.array([1, 1, 1])
        c2 = np.random.randn(n_per_cluster, 3) * 0.5 + np.array([8, 1, 2])
        c3 = np.random.randn(n_per_cluster, 3) * 0.5 + np.array([1, 8, 2])
        c4 = np.random.randn(n_per_cluster, 3) * 0.5 + np.array([8, 8, 1])
        data_3d = np.vstack([c1, c2, c3, c4])
        np.savetxt("puntos.txt", data_3d, fmt="%.2f")
    elif tipo== '2D':
        n = 200
        data = np.random.rand(n, 2) * 10
        np.savetxt("puntos.txt", data, fmt="%.2f")
        print("Archivo 'puntos.txt' generado con", n, "puntos.")
    elif tipo == '5D':
        n_per_cluster = 30
        c1 = np.random.randn(n_per_cluster, 5) * 0.5 + np.array([1,1,1,1,1])
        c2 = np.random.randn(n_per_cluster, 5) * 0.5 + np.array([8,1,2,2,1])
        c3 = np.random.randn(n_per_cluster, 5) * 0.5 + np.array([1,8,2,1,3])
        c4 = np.random.randn(n_per_cluster, 5) * 0.5 + np.array([8,8,1,3,2])
        data_5d = np.vstack([c1, c2, c3, c4])
        np.savetxt("puntos.txt", data_5d, fmt="%.2f")
    k =4
    archivo = "puntos.txt"
    data = np.loadtxt(archivo)
    centroids, labels = k_means(data, k)
    print("\nCentroides finales:")
    print(centroids)
    print("\nAsignación de cada punto:")
    for i, punto in enumerate(data):
        print(f"Punto {punto} -> Cluster {labels[i]}")
    graficar(data, centroids, labels)
if __name__ == "__main__":
    main()
