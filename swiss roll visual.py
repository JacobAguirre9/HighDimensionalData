# This is code for the swiss roll implementation of Isomap
# All figures taken were used in the 3D plot, so they aren't exactly reproducible.
# You can, however, move the plot around if you run the code in python and can find the same image dimensions

print("Hello World")


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
# for MDS dimensionality reduction we import the above^



class Isomap:
	def __init__(self, k_neigh = 10):
		self.k_neigh = k_neigh	#Number of NearestNeighbors

	def graph(self, D):
		# D is a matrix such that it contains all the pair-wise distances
		# graph will simply show us the graphical representation

		N, N = D.shape

		graph = {}

		#Extract closest NearestNeighbors to a given point
		for pnt in range(N):

			NearestNeighbors = D[pnt].argsort()[:self.k_neigh]
			graph[pnt] = NearestNeighbors[:self.k_neigh]

		# We need to make sure that this is a directed graph, not undirected
		# Consider the classic example of traveling salesman problem and Dijkstra's algorithm as
		# The intuition for this reasoning
		# So, we have that

		for i in graph:
			for j in graph[i]:
				if np.any(graph[j] == i):
					continue
				else:
					graph[j] = np.append(graph[j], i)

		# Now, we need to generate a full graph matrix with the parameters we're wanting to find
		FullGraphMatrix = np.full((N,N), np.inf)

		for i in range(N):
			FullGraphMatrix[i][graph[i]] = D[i][graph[i]]

		return FullGraphMatrix

	def fit_transform(self, X):
		""" Embedds a data matrix with Isomap.
		:param X: Data Matrix with shape = (Ndim, Npts)
		:return emb: Embedding
		"""

		D = distance_matrix(X.T, X.T)

		G = self.graph(D)

		ShortestPathMatrix = shortest_path(csgraph = G, method="FW")
        # Here, we are choosing to use Floyd-Warshall, however you can use different algorithms, such as
		# Dijkstra's algorithm to find the shortest path between a and b

		# Now, we need to make sure that we modify this graph
		ShortestPathMatrix = ShortestPathMatrix ** 2
		ShortestPathMatrix *= - 0.5
		ShortestPathMatrix += - np.mean(ShortestPathMatrix, axis=0)

		emb = MDS(ShortestPathMatrix, 2)

		return emb



# Now, consider that we're wanting to use this isomap algorithm on the classic swiss roll example
# for this example, we need to generate a swiss roll in 3D
# So again, we import packages needed!


import matplotlib.pyplot as plt
import numpy as np

# Now, we need to make two functions for a 3D version and a 2D version
# I define X to be a dataset, such that the shape is equal to (Number of Dimensions, Number of points)
# and the parameter color is obvious


def SwissRollin3D(X, color):
	plt.style.use('ggplot')
	ax = plt.axes(projection='3d')
	ax.scatter3D(X[0], X[1], X[2], c = color, cmap="rainbow")
	ax.tick_params(labelleft=False, labelbottom=False, labelright=False, labeltop=False)
	plt.show()

	# Of course, if you want to visualize the plots, you will need to MANUALLY type in plt.show(SwissRollin3D)
	# OR manually type in plt.show(SwissRollin2D)
	# I turned this off, because when running in pycharm I didn't want the plots to generate every single time I tried to run the code


	# The rainbow color is my favorite as it really allows us to have a nice visualization of a swiss roll :D
def SwissRollin2D(X, color):
	plt.style.use('ggplot')
	plt.scatter(X[0], X[1], c=color, cmap='rainbow')
	plt.tick_params(labelleft=False, labelbottom=False, labelright=False, labeltop=False)
	plt.show()
	# Again, just to reiterate, simply insert function name and you will get the plot to appear

