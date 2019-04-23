#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt
import random
from math import e, log


LABELS_FILE = "labels"
IMAGES_FILE = "images"
TEST_IMAGES_FILE = "test_images"
TEST_LABELS_FILE = "test_labels"


g = lambda z: (1 / (1 + e**-z)) 

def get_data(images_file, labels_file, training_sets):
	''' Get data from MNIST handwritten digit images and labels 
	Parameters:
		training_sets: How many training sets to retrieve
	'''

	# Initialise y values as all zeros
	y = np.zeros((training_sets, 10))
	label = np.zeros(training_sets)

	with open(labels_file, "rb") as labels:
		# Skip the first data bytes
		labels.seek(8)
		# Counter denoting training set
		i = 0
		for l in np.fromfile(labels, dtype=np.uint8, count=training_sets):
			# Set y array to 1 for corresponding label
			y[i][l] = 1
			label[i] = l
			i += 1

	with open(images_file, "rb") as images:
		# Skip the first data bytes
		images.seek(12)
		# Get pixels and reshape into array
		x = np.reshape(np.fromfile(images, dtype=np.uint8, count=training_sets*784), (training_sets, 784))/255.
	
	return x, y, label



class NeuralNetwork:

	def __init__(self, examples=5000, learning_rate=0.4, regularisation_parameter=2):

		# Number of layers
		self.L = 3
		# Units in each layer (None is added to make indices more logical)
		self.s = (None, 784, 25, 10)
		# Number of output layers 
		self.K = 10

		# Number of training sets
		self.m = examples
		# Features
		self.n = 784
		# Regularisation parameter
		self.l = regularisation_parameter
		# Learning rate
		self.A = learning_rate
		# Number of iterations
		self.iteration = 0

		# Theta values initialised as random
		self.T1 = np.random.random((self.s[2], self.s[1]+1)) * 2 - 1 #np.load("NN.npz").values()[0]
		self.T2 = np.random.random((self.s[3], self.s[2]+1)) * 2 - 1 #np.load("NN.npz").values()[2] #

		# Images and labels from MNIST dataset put into X and Y variables
		self.X, self.Y, self.labels = get_data(IMAGES_FILE, LABELS_FILE, self.m)



	def train(self, until_cost=None, until_difference=None, until_iteration=None, status=True):

		while True:
			cost = self.J()

			if status:
				print "ITERATION #{} : {}".format(self.iteration, self.J())

			if until_cost:
				if cost <= until_cost:
					break

			if until_difference:
				pass

			if until_iteration:
				if self.iteration > until_iteration:
					break

			self.iterate()


	def test(self, examples=100, display=False):
		# Set number of training sets to test
		self.m_test = examples
		# Images and labels from MNIST dataset put into X and Y variables
		self.X_test, self.Y_test, self.label_test = get_data(TEST_IMAGES_FILE, TEST_LABELS_FILE, self.m_test)

		correct = 0
		for i in range(self.m_test):
			if np.argmax(self.h(~i)) == self.label_test[i]:
				correct += 1

		print " {}/{} ({}%) correct ".format(correct, self.m_test, 100.*correct/self.m_test)




	def h(self, t):
		''' Hypothesis function, returns output layer of neural network '''
		return self.activation_units(t)[3]


	def activation_units(self, t):
		''' Return array of each layer's activation units for training set '''

		# LAYER 1 (input layer)		
		# Add bias unit to training set input
		if t >= 0:
			x = np.insert(self.X[t], 0, 1)
		else:
			# Negative t indicates test training set
			x = np.insert(self.X_test[~t], 0, 1)
											
		# LAYER 2 (hidden layer)			
		# Sum up parameters with dot product
		z = np.dot(self.T1, x)					
		# Activation units for layer 2		
		a2 = g(z)							
		# Add bias units 					
		a2 = np.insert(a2, 0, 1)			

		# LAYER 3 (output layer)
		# Sum up parameters with dot product
		z = np.dot(self.T2, a2)
		# Activation units for layer 3 (output units)	
		a3 = g(z)

		# Return all activation units
		# None is added to make indices more logical
		return (None, x, a2, a3)


	def errors(self, t, a):
		''' Returns array of layer 2 and 3 errors for training set, t '''

		# Layer 3 errors
		e3 = a[3] - self.Y[t]

		# Layer 2 errors
		e2 = np.dot(self.T2.T, e3) * (a[2] * (1 - a[2]))

		# None is added to make indices more logical
		return (None, None, e2, e3)


	def J(self):
		''' Regularised cost function has two components:
		1. the hypothesis cost
		2. the parameter cost (for regularisation) '''


		# Sum for hypothesis cost
		h_sum = 0
		# Loop through training sets
		for t in range(self.m):
			# Hypothesis for training set
			h_i = self.h(t)

			for k in range(self.K):
				if self.Y[t][k] == 1:
					h_sum += -log(h_i[k])
				if self.Y[t][k] == 0:
					h_sum += -log(1 - h_i[k])

		# Sum for parameter cost
		p_sum = 0

		# Add regularisation cost of theta1 parameters
		for i in self.T1:
			# Slice to exclude bias unit
			for t in i[:-1]:
				p_sum += t**2
				

		# Add regularisation cost of theta2 parameters 
		for i in self.T2:
			# Slice to exclude bias unit
			for t in i[:-1]:
				p_sum += t**2

		return (h_sum/self.m) + self.l*p_sum/(2*self.m) 


	def gradient_check(self, D, gradients_amount=5):
		''' Print approximated values for parameter derivatives of T2
			and compare with calculated derivatives '''


		for i in range(gradients_amount):

			# Small change in theta
			q = 0.00001

			# Random indices
			x = random.randint(0, self.s[2]-1)
			y = random.randint(0, self.s[3]-1)

			# Underestimate of cost function
			self.T2[y][x] -= q
			J_under = self.J()

			# Overestimate of cost function
			self.T2[y][x] += 2*q
			J_over = self.J()

			# Back to original value
			self.T2[y][x] -= q

			# Approximate gradient
			approx_grad = (J_over - J_under) / (2*q)

			print "Approximate gradient =\t", approx_grad
			print "Calculated gradient =\t", D[y][x], "\n"


	def iterate(self, verbose=False):
		''' Calculates activation units, errors for nodes,
			Performs backpropagation to calculate parameter derivatives for one iteration
			Returns array of parameter derivatives for the iteration '''

		# Accumulator for parameter derivatives
		d1 = np.zeros_like(self.T1)
		d2 = np.zeros_like(self.T2)

		# Loop through training sets
		for t in range(self.m):

			a = self.activation_units(t)

			E = self.errors(t,a)

			# atleast_2d enables arrays to tranpose and perform dot product
			# E2 removes bias unit with slice
			E2 = np.atleast_2d(E[2][1:]).T
			# E3 is the error of last layer, no bias unit
			E3 = np.atleast_2d(E[3]).T

			a1 = np.atleast_2d(a[1])
			a2 = np.atleast_2d(a[2])
			
			# Parameter derivatives accumulation for layers 2 and 3
			d1 += np.dot(E2, a1)
			d2 += np.dot(E3, a2)

		# Uses matrix operations to exclude bias unit from being regularised
		# Creates a matrix of ones corresponding to parameters
		bias_unit_discriminator = np.ones_like(self.T1)
		# Set parameters involving bias unit = 0
		bias_unit_discriminator.T[-1] = np.zeros(self.s[2])
		# Average derivatives and regularise
		D1 = d1/self.m + self.l*self.T1*bias_unit_discriminator/(2*self.m)

		# Ditto
		bias_unit_discriminator = np.ones_like(self.T2)
		bias_unit_discriminator.T[-1] = np.zeros(self.s[3])
		D2 = d2/self.m + self.l*self.T2*bias_unit_discriminator/(2*self.m)


		# Update parameters
		self.T1 -= self.A * D1
		self.T2 -= self.A * D2

		self.iteration += 1

		return D2


	def display_guess(self, grid=4):
		''' Display the digit and neural network tries to guess it '''

		# Initialise figure
		fig = plt.figure()

		# Creates array of random ints corresponding to training sets
		examples = np.random.choice(self.m_test, grid**2, replace=False)

		for i in range(0, grid**2):
			# Hypothesis for training set
			h_i = np.argmax(self.h(~examples[i]))
			# Whether guess is correct
			correct = (self.label_test[examples[i]] == h_i)

			# Add tile to grid
			g = fig.add_subplot(grid, grid, i+1, frameon=True)
			g.set_axis_off()
			# Add title of image label
			g.set_title(h_i)
			# Reshape pixel array and show
			img = plt.imshow(np.reshape(self.X_test[examples[i]], (28,28)), cmap=("Greens" if correct else "Reds"))

		plt.show()


	def save(self):
		#dt = np.dtype([ ("LR", np.float64), ("RP", np.float64), ("T1", np.float64, self.T1.shape), ("T2", np.float64, self.T2.shape) ])
		#array = np.array(
		np.savez("NN", np.array([self.A,self.l]), self.T1, self.T2)

	def load(self):
		params, _T1, _T2 = np.load("NN")


if __name__ == '__main__':
	N = NeuralNetwork()
	N.train(until_iteration=100)
	N.J()
	N.test()
	N.display_guess()





'''
img = tset(0)[2]
plt.matshow(img)
plt.show()

Problems:
	overflow error in g(z)
	J() potential overflow error
	store hypotheses in variable



'''

'''      
      INPUT LAYER (1)          HIDDEN LAYER (2)        OUTPUT LAYER (3)

         /   ◯ (bias)               
        |    ◯                       /   ◯ (bias)
        |    ◯                      |    ◯
        |    ◯                      |    ◯                     
        |    ◯                      |    ◯                    /  ◯
        |    ◯                      |    ◯                   |   ◯
        |    ◯                      |    ◯                   |   ◯
        |    ◯                      |    ◯                   |   ◯
784+1   |   ...              25+1   |   ...            10    |   ◯
units   |    ◯              units   |    ◯            units  |   ◯
        |    ◯                      |    ◯                   |   ◯
        |    ◯                      |    ◯                   |   ◯
        |    ◯                      |    ◯                   |   ◯
        |    ◯                      |    ◯                    \  ◯
        |    ◯                      |    ◯
        |    ◯                       \   ◯
        |    ◯              
         \   ◯               



T2 - Parameters going from 25-unit Layer 2 to the 10-unit Layer 3  ( 10 x 26 array )

          											   Origin
			 																								     bias
              1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26
 D        ┌   																									      ┐
 e      1 │   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
 s      2 │   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
 t      3 │   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
 i      4 │   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
 n      5 │   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
 a      6 │   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
 t      7 │   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
 i      8 │   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
 o      9 │   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
 n     10 │   0   0   X   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   │
          └ 	 																									  ┘
                  X = T2[10][3] 


# Example output layer representation
# 1  [ 0.2134 ] 
# 2  [ 0.6345 ] 
# 3  [ 0.2224 ] 
# 4  [ 0.2064 ] 
# 5  [ 0.9554 ] 
# 6  [ 0.1124 ] 
# 7  [ 0.0067 ] 
# 8  [ 0.1264 ] 
# 9  [ 0.7734 ] 
# 10 [ 0.5561 ] 


'''
