import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time

# ==============================
# Funciones FIS Sugeno (tu código original)
# ==============================
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

# ==============================
# MAIN: Cargar dataset S&P 500
# ==============================
df = pd.read_csv("sp500.csv", parse_dates=['Date'])
df.sort_values('Date', inplace=True)

# Filtrar últimos 5 años
fecha_limite = df['Date'].max() - pd.DateOffset(years=5)
df = df[df['Date'] >= fecha_limite]

# Usaremos solo la columna 'Close' para predecir
data_y = df['Close'].values
n = len(data_y)
data_x = np.arange(n)  # usar índice de día como variable de entrada
data = np.vstack((data_x, data_y)).T

# Graficar precios históricos
plt.figure(figsize=(10,5))
plt.plot(df['Date'], data_y, label="Close")
plt.title("Precio S&P 500")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.legend()
plt.show()

# ==============================
# Entrenar modelos con distintos radios
# ==============================
radios = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
resultados = []

for Ra in radios:
    fis_tmp = fis()
    fis_tmp.genfis(data, Ra, plot=False)
    r_pred = fis_tmp.evalfis(np.vstack(data_x))
    ecm = mean_squared_error(data_y, r_pred)
    resultados.append((Ra, ecm, r_pred))

# Mejor modelo según ECM
mejor_Ra, mejor_ecm, mejor_pred = min(resultados, key=lambda x: x[1])
print(f"Mejor radio: {mejor_Ra} con ECM = {mejor_ecm:.4f}")

# Graficar comparación
plt.figure(figsize=(10,5))
plt.plot(df['Date'], data_y, label="Real")
plt.plot(df['Date'], mejor_pred, '--', label=f"Sugeno (Ra={mejor_Ra})")
plt.legend()
plt.title("S&P 500: real vs modelo Sugeno")
plt.show()

# Graficar ECM vs Radio
plt.figure()
plt.plot([r[0] for r in resultados], [r[1] for r in resultados], marker='o')
plt.xlabel("Radio (Ra)")
plt.ylabel("ECM")
plt.title("ECM vs Radio en Subclust + Sugeno")
plt.show()

# ==============================
# Sobremuestreo más rápido (solo 2× puntos)
# ==============================
# Crear vector de índices para sobremuestreo
data_x_super = np.linspace(0, n-1, n*2)

# Evaluar modelo Sugeno
fis_best = fis()
fis_best.genfis(data, mejor_Ra, plot=False)
r_super = fis_best.evalfis(np.vstack(data_x_super))

# Convertir índices sobremuestreados a fechas reales
fecha_inicio = df['Date'].iloc[0]
fecha_fin = df['Date'].iloc[-1]
fechas_super = pd.date_range(fecha_inicio, fecha_fin, periods=n*2)

plt.figure(figsize=(12,5))
plt.plot(df['Date'], data_y, 'o', markersize=3, alpha=0.6, label="Original")
plt.plot(fechas_super, r_super, '-', linewidth=2, label="Sobremuestreada Sugeno")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.title("S&P 500: Original vs Sobremuestreada")
plt.legend()
plt.show()


# ==============================
# Extrapolación a futuro
# ==============================
# Crear 30 días futuros
future_days = 30
data_x_future = np.arange(n, n+future_days)
r_future = fis_best.evalfis(np.vstack(data_x_future))

plt.figure(figsize=(12,5))
plt.plot(df['Date'], data_y, label="Histórico")
future_dates = pd.date_range(df['Date'].iloc[-1]+pd.Timedelta(days=1), periods=future_days)
plt.plot(future_dates, r_future, '--', color='red', label="Extrapolación")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.title("Predicción futura S&P 500 con Sugeno")
plt.legend()
plt.show()
