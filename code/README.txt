Name: Chirag Khurana
Entry No.: 2016CSB1037
CSL603 Lab3

==============================How to run====================================
Libraries Required:
	1. numpy
	2. scipy
		To install: sudo apt-get install python3-scipy
	3. opencv
		To install: http://cyaninfinite.com/tutorials/installing-opencv-in-ubuntu-for-python-3/
	4. matplotlib


To train model:
	$ python3 main.py
	To change parameters, change them initialized at begin of main.py:
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
		DROPOUT_RATE

	Note: It can work for different architectures just change ARCHITECTURE.

To plot graph after train:
	$ python3 graph.py

===============================END========================================