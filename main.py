# This is code for the creation of an algorithm for Isomap and Local linear embedding
# This code was developed for MATH 4803 taught by Dr. Liao at Georgia Institute of Technology

print("Hello World")


# For the Local linear mapping, I am planning to use the MNIST dataset, and compare
# using LLE & logistic regression vs logistic regression alone for accuracy

# First, I am loading the dataset from scikit
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
plt.show()





