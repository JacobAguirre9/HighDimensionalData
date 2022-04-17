# This is code for the creation of an algorithm for Isomap and Local linear embedding
# This code was developed for MATH 4803 taught by Dr. Liao at Georgia Institute of Technology

print("Hello World")

from sklearn.datasets import load_digits
digits = load_digits()
type(digits.data)

(digits.data.shape, digits.target.shape, digits.images.shape)
