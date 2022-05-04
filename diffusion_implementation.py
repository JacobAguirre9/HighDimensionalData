# %matplotlib widget
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def diffusion_matrix(X, epsilon=0.15, steps = 32):
  # return the symmetrized diffusion matrix P

  # X: nxp array
  # K: nxn matrix
  K = np.zeros((X.shape[0], X.shape[0]))
  for row in range(X.shape[0]):
    K[row] = np.linalg.norm(X[row]-X, axis = 1)
  # kernel matrix
  K = np.exp(-K**2 / epsilon)
  D = np.diag(np.sum(K, axis = 0))
  # connectivity matrix
  P = np.linalg.inv(D) @ K
  D_right = np.diag(np.sum(K, axis = 0)**0.5)
  D_left = np.diag(np.sum(K, axis = 0)**-0.5)
  #symmetrize connectivity matrix P
  P_sym = D_right @ (P@D_left)
  #Diffusion process
  P_sym = np.linalg.matrix_power(P_sym, steps)

  return P_sym, D_left

def diffusion_coor(P_sym, D_left, k=3):

    e, v = np.linalg.eigh(P_sym)
    indices = e.argsort()[::-1]
    v = v[:, indices]
    diffusion_coordinates = D_left @ v
    
    return diffusion_coordinates[:,:k]

length_phi = 18   #length of swiss roll in angular direction
length_Z = 15     #length of swiss roll in z direction
sigma = 0.001       #noise strength
m = 10000         #number of samples



# create dataset
phi = length_phi*np.random.rand(m)
xi = np.random.rand(m)
Z = length_Z*np.random.rand(m)
X = 1./6*(phi + sigma*xi)*np.sin(phi)
Y = 1./6*(phi + sigma*xi)*np.cos(phi)
col = np.arange(m)

swiss_roll = np.array([X, Y, Z]).transpose()

P_sym, D_left = diffusion_matrix(swiss_roll, epsilon = 0.3, steps=2**10)
coor = diffusion_coor(P_sym, D_left, k=2)


fig = plt.figure(0)
ax = Axes3D(fig)
ax.scatter(X,Y,Z,c=phi)
plt.figure(1)
plt.scatter(coor[:,0], coor[:,1], c=phi,s=15)
plt.show()
