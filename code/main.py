import numpy as np
import random
import os
from scipy.special import expit as sig
import cv2
import matplotlib.pyplot as plt
import pickle

# Dataset folder
DATASET_DIRECTORY = '../steering/'
# Training/validation split ratio
SPLIT_RATIO = 0.8
# Architecture
ARCHITECTURE = [1024,512,64,1]
# Minibatch Size
MINIBATCH_SIZE = 64
# Learning Rate
LEARNING_RATE = 0.01
# No. of epochs
EPOCHS = 1000
# Dropout rate
DROPOUT_RATE = 0


def read_data(dataset_dir,split_ratio):
	# Reading data.txt
	file = open(os.path.join(dataset_dir,'data.txt'))
	line = file.readline()

	# Data
	X = []
	Y = []
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []

	while line:
		[img_path, deg] = line.split('\t')

		img_path = os.path.join(dataset_dir,img_path)
		# removing '\n' from deg
		deg = float(deg[:-1])
		img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
		
		if img is None:
			line = file.readline()
			continue

		# Normalization
		img = (img - img.min())/img.max()

		X.append(img.flatten().T)
		Y.append(deg)
		line = file.readline()

	file.close()
	# Splitting data
	n = len(Y)
	r = random.sample(range(0, n), n)
	# Training data
	for i in r[:round(split_ratio*n)]:
		X_train.append(X[i])
		Y_train.append(Y[i])
	# Testing data
	for i in r[round(split_ratio*n):]:
		X_test.append(X[i])
		Y_test.append(Y[i])

	X_train = np.column_stack(tuple(X_train))
	Y_train = np.array(Y_train)
	X_test = np.column_stack(tuple(X_test))
	Y_test = np.array(Y_test)

	return X_train, Y_train, X_test, Y_test


class Neural_Network:
	
	def __init__(self, architecture):
		self.architecture = architecture
		self.n_layers = len(architecture)-1
		
		# Initializing random weights between -0.01 to 0.01
		self.weights = []
		for i in range(self.n_layers):
			# +1 for bias e.g. 1025x512
			w = np.zeros((architecture[i]+1,architecture[i+1]))
			for j in range(architecture[i]):
				w[j,:] = np.linspace(-0.01, 0.01, num=architecture[i+1]);
			# w[1:,:] = (np.random.rand(architecture[i],architecture[i+1]) - 0.5)/50
			self.weights.append(w.T)



	def train_network(self, X_train, Y_train, X_test, Y_test, mini_batch_size, epochs, learning_rate, dropout_rate=0):
		N = len(Y_train)

		# Data shuffling
		# r = random.sample(range(0, N), N)
		# X_train = X_train[:,r]
		# Y_train = Y_train[r]

		# For plotting graph
		self.X_Graph = []
		self.Y_train_Graph = []
		self.Y_test_Graph = []
		# Running epochs
		for i in range(epochs):
			# For each minibatch
			for j in range(0, N, mini_batch_size):
				N_mini = min(j+mini_batch_size, N) - j
				X_mini = X_train[:,j:j+mini_batch_size]
				# Adding bias term
				X_mini = np.row_stack((np.ones((1, N_mini)), X_mini))
				Y_mini = Y_train[j:j+mini_batch_size]
				
				# Forward Pass
				# Z will store input, and output of every layer
				Z = [X_mini]
				for k in range(self.n_layers):
					# weight for kth layer
					w = self.weights[k]

					if dropout_rate>0 and dropout_rate<1:
						# Bias term is not effected
						Z[k][1:,] *= np.tile(np.random.binomial(1, 1 - dropout_rate, (Z[k].shape[0] - 1, 1)), mini_batch_size)
						
					# Output of kth layer Z[k+1] = w*Z[k]
					zk = w.dot(Z[k])
					# For last layer there is no activation function
					if k != self.n_layers-1:
						# Activation function: Sigmoid
						zk = sig(zk)
						# Adding bias term
						zk = np.row_stack((np.ones((1, N_mini)), zk))

					Z.append(zk)

				# Backward Pass
				delta_weights = []
				for k in range(self.n_layers):
					delta_weights.append(np.zeros((self.architecture[k]+1,self.architecture[k+1])).T)

				# For last layer as it don't has activation function
				delta_z = Z[self.n_layers]-Y_mini
				# delta_w = (output - y)*Z(K-1)
				# delta_weights[self.n_layers-1] = delta_z.dot(Z[self.n_layers-1].T)/N_mini
				delta_weights[self.n_layers-1] = delta_z.dot(Z[self.n_layers-1].T)

				for k in range(self.n_layers-2,-1,-1):
					delta_z = Z[k+1]*(1-Z[k+1])*self.weights[k+1].T.dot(delta_z)
					# Removing bias term
					delta_z = delta_z[1:,:]

					# delta_weights[k] = delta_z.dot(Z[k].T)/N_mini
					delta_weights[k] = delta_z.dot(Z[k].T)

				for k in range(self.n_layers):
					self.weights[k] = self.weights[k] - learning_rate*delta_weights[k]

			training_mse = self.MSE(X_train,Y_train)
			testing_mse = self.MSE(X_test, Y_test)
			print(i+1,"\tTraining "+str(training_mse)+"\tTesting "+str(testing_mse))
			self.X_Graph.append(i+1)
			self.Y_train_Graph.append(training_mse)
			self.Y_test_Graph.append(testing_mse)



	def MSE(self,X_test, Y_test):
		# No of instances
		N = len(Y_test)
		# Adding bias 1s
		X_test = np.row_stack((np.ones((1,N)),X_test))
		# Z will store output of every layer
		Z = X_test
		for k in range(self.n_layers):
			# weight for kth layer
			w = self.weights[k]

			zk = w.dot(Z)
			# For last layer there is no activation function
			if k != self.n_layers-1:
				# Activation function: Sigmoid
				zk = sig(zk)
				# Adding bias term
				zk = np.row_stack((np.ones((1, N)), zk))

			Z = zk

		return np.sum((Z - Y_test)**2)/N





X_train, Y_train, X_test, Y_test = read_data(DATASET_DIRECTORY,SPLIT_RATIO)

ann = Neural_Network(ARCHITECTURE)
ann.train_network(X_train,Y_train,X_test,Y_test,MINIBATCH_SIZE,EPOCHS,LEARNING_RATE,DROPOUT_RATE)
outfile = open('X_train','wb')
pickle.dump(ann.X_Graph,outfile)
outfile.close()
outfile = open('Y_train','wb')
pickle.dump(ann.Y_train_Graph,outfile)
outfile.close()
outfile = open('Y_test','wb')
pickle.dump(ann.Y_test_Graph,outfile)
outfile.close()