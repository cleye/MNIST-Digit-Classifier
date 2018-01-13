#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from math import e, log


LABELS_FILE = "/Users/cleye.jensen/Downloads/labels"
IMAGES_FILE = "/Users/cleye.jensen/Downloads/images"
TEST_IMAGES_FILE = "/Users/cleye.jensen/Downloads/test_images"
TEST_LABELS_FILE = "/Users/cleye.jensen/Downloads/test_labels"


# Number of layers
L = 3
# Units in each layer
s = (None, 784, 25, 10)
# Number of output layers (None is added to make indices more logical)
K = 10

# Number of training sets
m = 5000
m_test = 100
# Features
n = 784
# Regularisation parameter
l = 1
# Learning rate
A = 0.05
# Number of iterations
iteration = 0


# Theta values initialised as random in bound [-10, 10]
T1 = np.random.random((s[2], s[1]+1)) * 0.2 - 0.1
T2 = np.random.random((s[3], s[2]+1)) * 2 - 1



def get_data(images_file, labels_file, training_sets):
	''' Get data from MNIST handwritten digit images and labels 
	Parameters:
		training_sets: How many training sets to retrieve
	'''

	# Initialise y values as all zeros
	y = np.zeros((training_sets,K))

	with open(labels_file, "rb") as labels:
		# Skip the first data bytes
		labels.seek(8)
		# Counter denoting training set
		i = 0
		for l in np.fromfile(labels, dtype=np.uint8, count=training_sets):
			# Set y array to 1 for corresponding label
			y[i][l] = 1
			i += 1

	with open(images_file, "rb") as images:
		# Skip the first data bytes
		images.seek(12)
		# Get pixels and reshape into array
		x = np.reshape(np.fromfile(images, dtype=np.uint8, count=training_sets*784), (training_sets, 28, 28))/255.
	
	return x,y

X,Y = get_data(IMAGES_FILE, LABELS_FILE, m)
X_test, Y_test = get_data(TEST_IMAGES_FILE, TEST_LABELS_FILE, m_test)

def g(z):
	''' Sigmoid function '''
	return (1  / (1 + e**-z)) 


def h(t):
	''' Hypothesis function, returns output layer of neural network '''
	return activation_units(t)[3]


def activation_units(t):
	''' Return array of each layer's activation units for training set '''

	# LAYER 1 (input layer)		
	# Add bias unit to training set input
	if t >= 0:
		x = np.insert(X[t], 0, 1)
	else:
		# Negative t indicates test training set
		x = np.insert(X_test[t], 0, 1)
										
	# LAYER 2 (hidden layer)			
	# Sum up parameters with dot product
	z = np.dot(T1, x)					
	# Activation units for layer 2		
	a2 = g(z)							
	# Add bias units 					
	a2 = np.insert(a2, 0, 1)			

	# LAYER 3 (output layer)
	# Sum up parameters with dot product
	z = np.dot(T2, a2)
	# Activation units for layer 3 (output units)	
	a3 = g(z)

	# Return all activation units
	# None is added to make indices more logical
	return (None, x, a2, a3)


def errors(t,a):
	''' Returns array of layer 2 and 3 errors for training set, t '''

	# Layer 3 errors
	e3 = a[3] - Y[t]

	# Layer 2 errors
	e2 = np.dot(T2.T, e3) * (a[2] * (1 - a[2]))

	# None is added to make indices more logical
	return (None, None, e2, e3)


def J():
	''' Regularised cost function has two components:
	1. the hypothesis cost
	2. the parameter cost (for regularisation) '''


	# Sum for hypothesis cost
	h_sum = 0
	# Loop through training sets
	for t in range(m):
		# Hypothesis for training set
		h_i = h(t)

		for k in range(K):
			if Y[t][k] == 1:
				h_sum += -log(h_i[k])
			if Y[t][k] == 0:
				h_sum += -log(1 - h_i[k])

	# Sum for parameter cost
	p_sum = 0

	# Add regularisation cost of theta1 parameters
	for i in T1:
		# Slice to exclude bias unit
		for t in i[:-1]:
			p_sum += t**2
			

	# Add regularisation cost of theta2 parameters 
	for i in T2:
		# Slice to exclude bias unit
		for t in i[:-1]:
			p_sum += t**2

	return (h_sum/m) + l*p_sum/(2*m) 


def gradient_check(D, amount=5):
	''' Print approximated values for parameter derivatives of T2
		and compare with calculated derivatives '''

	# Amount of gradients to check
	gradients_amount = amount

	for i in range(gradients_amount):

		# Small change in theta
		q = 0.00001

		# Random indices
		x = random.randint(0, s[2]-1)
		y = random.randint(0, s[3]-1)

		# Underestimate of cost function
		T2[y][x] -= q
		J_under = J()

		# Overestimate of cost function
		T2[y][x] += 2*q
		J_over = J()

		# Back to original value
		T2[y][x] -= q

		# Approximate gradient
		approx_grad = (J_over - J_under) / (2*q)

		print "Approximate gradient =\t", approx_grad
		print "Calculated gradient =\t", D[y][x], "\n"


def iterate(verbose=False):
	''' Calculates activation units, errors for nodes,
		Performs backpropagation to calculate parameter derivatives for one iteration
		Returns array of parameter derivatives for the iteration '''

	global T1, T2, iteration

	# Accumulator for parameter derivatives
	d1 = np.zeros_like(T1)
	d2 = np.zeros_like(T2)

	# Loop through training sets
	for t in range(m):

		a = activation_units(t)

		E = errors(t,a)

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
	bias_unit_discriminator = np.ones_like(T1)
	# Set parameters involving bias unit = 0
	bias_unit_discriminator.T[-1] = np.zeros(s[2])
	# Average derivatives and regularise
	D1 = d1/m + l*T1*bias_unit_discriminator/(2*m)

	# Ditto
	bias_unit_discriminator = np.ones_like(T2)
	bias_unit_discriminator.T[-1] = np.zeros(s[3])
	D2 = d2/m + l*T2*bias_unit_discriminator/(2*m)


	# Update parameters
	T1 -= A * D1
	T2 -= A * D2

	iteration += 1

	return D2


def status(percentage_correct=False, fancy=False):
	''' Prints status of neural net such as cost and images correct 
	Parameters:
		percentage_correct: 
		fancy: Whether to use fancy formatting for status
	'''
	if fancy:
		print "-"*16
		print "ITERATION #{}".format(iteration)
		print "Cost: \t", J()
		print "-"*16

	if not fancy:
		print "ITERATION #{} : {}".format(iteration,J())

	if percentage_correct:
		test_amount = 100
		correct = 0
		# Test if hypotheses is same as actual value
		for i in range(test_amount):
			if np.argmax(h(~i)) == np.argmax(Y_test[i]):
				correct += 1

		print correct, "/100 ({}%) correct".format(100*correct/test_amount)



np.set_printoptions(threshold=1000000,suppress=True)

def run():
	for i in range(6):
		for i in range(49):
			iterate()
			status()
		iterate()
		status(percentage_correct=True, fancy=True)

#run()


'''
img = tset(0)[2]
plt.matshow(img)
plt.show()

Problems:
	overflow error in g(z)
	second layer units are 1s or 0s
	J() potential overflow error

'''

'''      
      INPUT LAYER (1)         HIDDEN LAYER (2)        OUTPUT LAYER (3)

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

