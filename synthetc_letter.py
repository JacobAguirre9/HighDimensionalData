 
# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pydiffmap import diffusion_map as dm
from pydiffmap.visualization import embedding_plot, data_plot

x = np.arange(-200,200) + np.random.normal(0, 20, 400)
y = 2*np.abs(x)+ np.random.normal(0, 20, 400)
z = np.random.normal(0, 50, 400)
  
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

data = np.zeros((400, 3))
data[:,0] = x
data[:,1] = y
data[:,2] = z

neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}

mydmap = dm.DiffusionMap.from_sklearn(n_evecs=2, k=256, epsilon='bgh', alpha=1, neighbor_params=neighbor_params)
# fit to data and return the diffusion map.
dmap = mydmap.fit_transform(data)
embedding_plot(mydmap, scatter_kwargs = {'c': dmap[:,0], 'cmap': 'Spectral'})
data_plot(mydmap, dim=3, scatter_kwargs = {'cmap': 'Spectral'})

# plt.show()
# displaying the plot
plt.show()