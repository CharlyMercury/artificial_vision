import numpy as np
import matplotlib.pyploy as plt
from sklearn.datasets import make_gaussian_quantiles

# Creemos datasets desde cero - Para un ejemplo de clasificacion
N = 1000
gaussian_quantiles = make_gaussian_quantiles(
	mean=None,
	cov=0.1,
	n_samples=N,
	n_features=2,
	n_classes=2,
	shuffle=True,
	random_state=None)

X, Y = gaussian_quantiles

print(X.shape)
print(Y.shape)

plt.scatter(X[:,0], X[:,1],c=Y[:,0], s=40, cmap=pĺt.cm.spectral)
