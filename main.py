#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from math import e, log


class TrainingSet:
	''' Training set from MNIST handwritten digit images

	Attributes:
		index: An integer index of the training set within the image file
		y: A numpy column array of the desired neural net output
		x: A numpy array of pixels in the training set image
		image_2d: A numpy matrix of pixels in the training set image

	'''

	def __init__(self, index_):
		pass
		''' Inits TrainingSet by reading training set data '''
		'''
		self.LABELS = "/Users/cleye.jensen/Downloads/labels"
		self.IMAGES = "/Users/cleye.jensen/Downloads/images"

		self.index = index_

		with open(self.LABELS, "rb") as labels:
			# Position cursor at training set
			labels.seek(self.index)
			# Read 1 byte and convert to int
			self.label = int.from_bytes(labels.read(1), byteorder="big")

		# Set expected output from neural net
		self.y = np.zeros(10)
		self.y[self.label] = 1

		# improve this
		with open(self.IMAGES, "rb") as images:
			# Position cursor at training set
			images.seek(self.index * 784)
			# 1D array of pixels
			image = []

			# Loop through rows and columns of image pixels
			for i in range(28):
				for i in range(28):
					# Read 1 byte and convert to int
					pixel = 1#int.from_bytes(images.read(1), byteorder="big")
					image.append(pixel)

		self.x = np.array(image)/255.

	def display():'''
		''' Displays the training set image '''
		#plt.matshow(img)
		#plt.show()




# Number of layers
L = 3
# Units in each layer
s = (None, 784, 25, 10)
# Number of output layers (None is added to make indices more logical)
K = 10

# Training sets
m = 1000
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
T2 = np.random.random((s[3], s[2]+1)) * 0.2 - 0.1


def g(z):
	''' Sigmoid function '''
	return (1  / (1 + e**-z)) 


def h(t):
	''' Hypothesis function, returns output layer of neural network '''
	return activiation_units(t)[3]


def activiation_units(t):
	''' Return array of each layer's activation units for training set '''

	# LAYER 1 (input layer)						# Example output layer representation
	# Add bias unit to training set input			# 1  [ 0.2134 ] 
	x = np.insert(t.x, 0, 1)						# 2  [ 0.6345 ] 
													# 3  [ 0.2224 ] 
	# LAYER 2 (hidden layer)						# 4  [ 0.2064 ] 
	# Sum up parameters with dot product			# 5  [ 0.9554 ] 
	z = np.dot(T1, x)								# 6  [ 0.1124 ] 
	# Activation units for layer 2					# 7  [ 0.0067 ] 
	a2 = g(z)										# 8  [ 0.1264 ] 
	# Add bias units 								# 9  [ 0.7734 ] 
	a2 = np.insert(a2, 0, 1)						# 10 [ 0.5561 ] 

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
	e3 = a[3] - t.y

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
	for i in range(m):

		t = TrainingSet(i)
		# Hypothesis for training set
		h_i = h(t)

		for k in range(K):
			if t.y[k] == 1:
				h_sum += -log(h_i[k])
			if t.y[k] == 0:
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


def gradient_check(D):
	''' Print approximated values for parameter derivatives of T2
		and compare with calculated derivatives '''

	# Amount of gradients to check
	gradients_amount = 5

	for i in range(gradients_amount):

		# Small change in theta
		q = 0.0001

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

	global T1,T2, iteration

	# Accumulator for parameter derivatives
	d1 = np.zeros_like(T1)
	d2 = np.zeros_like(T2)

	# Loop through training sets
	for i in range(m):

		t = TrainingSet(i)

		a = activiation_units(t)

		E = errors(t,a)

		# atleast_2d enables arrays to tranpose and perform dot product
		# E2 removes bias unit with slice
		E2 = np.atleast_2d(E[2][1:]).T
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


def status():
	#print "-"*16
	print "ITERATION #{} : {}".format(iteration,J())
	#print "Cost: \t", J()
	#print ""
	#print "-"*16

np.set_printoptions(threshold=1000000,suppress=True)

def run():
	iterate()
	status()
	D = iterate()
	gradient_check(D)


labels = np.fromfile("/Users/cleye.jensen/Downloads/labels", dtype=np.uint8)
images = np.reshape(np.fromfile("/Users/cleye.jensen/Downloads/images", dtype=np.uint8, count=m*784), (m, 28, 28))






'''
img = tset(0)[2]
plt.matshow(img)
plt.show()



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


'''

