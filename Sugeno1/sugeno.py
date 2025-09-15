import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time

def subclust2(data, Ra, Rb=0, AcceptRatio=0.3, RejectRatio=0.1, plot=False):
    if Rb==0:
        Rb = Ra*1.15

    scaler = MinMaxScaler()
    scaler.fit(data)
    ndata = scaler.transform(data)

    P = distance_matrix(ndata,ndata)
    alpha=(Ra/2)**2
    P = np.sum(np.exp(-P**2/alpha),axis=0)

    centers = []
    i=np.argmax(P)
    C = ndata[i]
    p=P[i]
    centers = [C]

    continuar=True
    restarP = True
    while continuar:
        pAnt = p
        if restarP:
            P=P-p*np.array([np.exp(-np.linalg.norm(v-C)**2/(Rb/2)**2) for v in ndata])
        restarP = True
        i=np.argmax(P)
        C = ndata[i]
        p=P[i]
        if p>AcceptRatio*pAnt:
            centers = np.vstack((centers,C))
        elif p<RejectRatio*pAnt:
            continuar=False
        else:
            dr = np.min([np.linalg.norm(v-C) for v in centers])
            if dr/Ra+p/pAnt>=1:
                centers = np.vstack((centers,C))
            else:
                P[i]=0
                restarP = False
        if not any(v>0 for v in P):
            continuar = False

    distancias = [[np.linalg.norm(p-c) for p in ndata] for c in centers]
    labels = np.argmin(distancias, axis=0)
    centers = scaler.inverse_transform(centers)

    # 👇 Bloque opcional para graficar
    if plot:
        plt.figure()
        plt.scatter(data[:,0], data[:,1], c=labels, cmap='viridis', alpha=0.6)
        plt.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=200, edgecolors='black')
        plt.title("Clusters encontrados con Subclust")
        plt.show()

    return labels, centers

def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

class fisRule:
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma

class fisInput:
    def __init__(self, min,max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids

    def view(self):
        x = np.linspace(self.minValue,self.maxValue,20)
        plt.figure()
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
            plt.plot(x,y)

class fis:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []

    def genfis(self, data, radii, plot=False):
        start_time = time.time()
        labels, cluster_center = subclust2(data, radii, plot=plot)
        print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)

        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        A = acti*inp/sumMu
        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions
        return 0
    
    def evalfis(self, data):
        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)

    def viewInputs(self):
        for input in self.inputs:
            input.view()

# =============================
# MAIN: Cargar datos
# =============================
data_y = np.loadtxt("samplesVDA4.txt")   # un valor por línea
n = len(data_y)
data_x = np.arange(0, n*0.025, 0.025)
data = np.vstack((data_x, data_y)).T

# =============================
# Probar varios radios
# =============================
radios = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]
resultados = []

for Ra in radios:
    fis_tmp = fis()
    fis_tmp.genfis(data, Ra, plot=False)   # plot=True si querés ver clusters
    r_pred = fis_tmp.evalfis(np.vstack(data_x))
    ecm = mean_squared_error(data_y, r_pred)
    resultados.append((Ra, ecm, r_pred))

# Elegir el mejor
mejor_Ra, mejor_ecm, mejor_pred = min(resultados, key=lambda x: x[1])
print(f" Mejor radio: {mejor_Ra} con ECM = {mejor_ecm:.4f}")

# =============================
# Graficar comparación
# =============================
plt.figure()
plt.plot(data_x, data_y, label="Medido")
plt.plot(data_x, mejor_pred, '--', label=f"FIS Sugeno (Ra={mejor_Ra})")
plt.legend()
plt.show()

# =============================
# Graficar ECM vs Radio
# =============================
plt.figure()
plt.plot([r[0] for r in resultados], [r[1] for r in resultados], marker='o')
plt.xlabel("Radio de cluster (Ra)")
plt.ylabel("Error cuadrático medio (ECM)")
plt.title("Optimización de radio en Subclust + Sugeno")
plt.show()

# =============================
# Sobremuestreo
# =============================

# Crear un vector de tiempo con paso más chico (ej: 0.005s en vez de 0.025s)
data_x_super = np.arange(0, n*0.025, 0.005)

# Evaluar el modelo Sugeno en esos puntos
fis_best = fis()
fis_best.genfis(data, mejor_Ra, plot=False)
r_super = fis_best.evalfis(np.vstack(data_x_super))

# =============================
# Graficar original vs sobremuestreada
# =============================
plt.figure(figsize=(10,5))
plt.plot(data_x, data_y, 'o', markersize=3, alpha=0.6, label="Señal original (0.025s)")
plt.plot(data_x_super, r_super, '-', linewidth=2, label="Sobremuestreada con Sugeno (0.005s)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Diámetro arterial")
plt.legend()
plt.title("Señal original vs sobremuestreada con Sugeno")
plt.show()
