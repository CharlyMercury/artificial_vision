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

plt.scatter(X[:,0], X[:,1],c=Y, s=40, cmap=pĺt.cm.spectral)

# Funciones de activacion

def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x)/(np.exp(-x)+1)**2
    else:
        return 1/(1+np.exp(-x))

def relu(x, derivate=False):
    if derivate:
        x[x<=0]=0
        x[x>0]=1
        return x
    else:
        return np.maximun(0,x)

# Funcion de perdida
def mse(y, y_hat, derivate=False):
"""
    Esta función retorna el error cuadrático medio.
    Si derivada es True, regresa la derivada en un punto.

    :params: (float): 
    :params: (float): 

"""
    if derivate:
        return (y_hat-y)
    else:
        return np.mean((y_hat-y)**2)

# Estructura de la red: asignar pesos y biases

def initialize_parameters_deep(layers_dim: list):
    parameters = {}
    L = len(layers_dim)
    for l in range(0,L-1):
        parameters['W'+str(l+1)] = (np.random.rand(
                        layers_dim[l], layers_dim[l+1])*2)-1
        parameters['b'+str(l+1)] = (np.random.rand(1, 
                        layers_dim[l+1])*2)-1
    return parameters


layers_dims = [2, 4, 8, 1]

params = initialize_parameters_deep(layers_dims)
print(params)

params['A0']=X

# Primer capa
params['Z1']=np.matmul(params['A0'], params['W1'])+params['b1']
params['A1']=relu(params['Z1'])

# Segunda capa
params['Z2']=np.matmul(params['A1'], params['W2'])+params['b2']
params['A2']=relu(params['Z2'])

# Tercer capa
params['Z3']=np.matmul(params['A2'], params['W3'])+params['b3']
params['A3']=sigmoid(params['Z3'])

output = params['A3']


# Backpropagation

# Pesos en la última capa
params['dZ3']=mse(Y, output, True)*sigmoid(params['A3'], True)
params['dW3']=np.matmul(params['A2'].T, params['dZ3'])

# Pesos en la penúltima capa
params['dZ2']=np.matmul(params['dZ3'], params['W3'].T)*relu(params['A2', True])
params['dW2']=np.matmul(params['A1'].T, params['dZ2'])

# Pesos de la primer capa
params['dZ1']=np.matmul(params['dZ2'], params['W2'].T)*relu(params['A1', True])
params['dW1']=np.matmul(params['A0'].T, params['dZ1'])


## Algoritmo del gradiente descendiente
learning_rate = 0.0001
params['W3']=params['W3']-params['dW3']*learing_rate
params['W2']=params['W2']-params['dW2']*learing_rate
params['W1']=params['W1']-params['dW1']*learing_rate

params['b3']=params['b3']-(np.mean(params['dW3'], axis=0, keepdims=True))*learning_rate
params['b2']=params['b2']-(np.mean(params['dW2'], axis=0, keepdims=True))*learning_rate
params['b1']=params['b1']-(np.mean(params['dW1'], axis=0, keepdims=True))*learning_rate




