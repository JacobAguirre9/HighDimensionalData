 
# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pydiffmap import diffusion_map as dm
from pydiffmap.visualization import embedding_plot, data_plot
from sklearn.neighbors import KNeighborsClassifier
samples = 500
theta = np.arange(0, 2*np.pi, 2*np.pi/samples)
r = np.array([5,10,15,20])
x = np.zeros((samples*len(r),))
y = np.zeros((samples*len(r),))
for i in range(len(r)):
    x[i*samples:(i+1)*samples] = r[i]*np.cos(theta) + np.random.normal(0, 0.1, samples)
    y[i*samples:(i+1)*samples] = r[i]*np.sin(theta) + np.random.normal(0, 0.1, samples)
z = np.random.normal(0, 0.01, samples*len(r))
  
# # creating figure
# fig = plt.figure()
# ax = Axes3D(fig)
  
# # creating the plot
# plot_geeks = ax.scatter(x, y, z, color='Green')
  
# # setting title and labels
# ax.set_title("3D plot")
# ax.set_xlabel('x-axis')
# ax.set_ylabel('y-axis')
# ax.set_zlabel('z-axis')

data = np.zeros((samples * len(r), 3))
data[:,0] = x
data[:,1] = y
data[:,2] = z

neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}

mydmap = dm.DiffusionMap.from_sklearn(n_evecs=2, k=200, epsilon='bgh', alpha=0.8, neighbor_params=neighbor_params)
# fit to data and return the diffusion map.
dmap = mydmap.fit_transform(data)
embedding_plot(mydmap, scatter_kwargs = {'c': dmap[:,1], 'cmap': 'Accent'})
data_plot(mydmap, dim=3, scatter_kwargs = {'cmap': 'Accent'})



plt.show()