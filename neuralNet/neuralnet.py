import sys
import scipy.io.arff as sciarff
import numpy as np
import random
import math
import pylab


def split(dataset, n):
	l = len(dataset)
	l_p = l/n
	length = [l_p for i in range(n)]
	t = l - n*l_p
	for i in range(t):
		length[i] += 1
	d1 = [t for t in dataset if t[-1] == 1]
	d0 = [t for t in dataset if t[-1] == 0]
	random.shuffle(d1)
	random.shuffle(d0)
	d = d1 + d0
	data_batch = [[] for i in range(n)]
	m = 0 
	for i in range(l):
		data_batch[m].append(d[i])
		m += 1
		if m == n:
			m = 0
	return data_batch

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x) * (1 - sigmoid(x))

def column(matrix, i):
    return [[row[i]] for row in matrix]


def plot_learning_curve(epoches):
	accuracy_epoch = [[] for i in range(len(epoches))]
	for i in range(len(epoches)):
		net = Network([len(features)-1, len(features)-1, 1])
		accuracy_epoch[i], actual, confi = net.sgd_cross(train, 0.1, epoches[i], 10)
	pylab.figure(1)
	pylab.plot(epoches, accuracy_epoch, 'b', label = 'accuracy')
	pylab.title("Accuracies for different number of epoches")
	pylab.xlabel("Number of epoches")
	pylab.ylabel("test-set accuracy")
	pylab.show()

def plot_folds(folds):
	accuracy_folds = [[] for i in range(len(folds))]
	for i in range(len(folds)):
		net = Network([len(features)-1, len(features)-1, 1])
		accuracy_folds[i], actual, confi = net.sgd_cross(train, 0.1, 50, folds[i])
	pylab.figure(2)
	pylab.plot(folds, accuracy_folds, 'b', label = 'accuracy')
	pylab.title("Accuracies for different number of folds")
	pylab.xlabel("Number of folds")
	pylab.ylabel("test-set accuracy")
	pylab.show()

def plot_roc(num_epochs, learning_rate, num_folds):
	res = []
	net = Network([len(features)-1, len(features)-1, 1])
	t, actual, confi = net.sgd_cross(train, learning_rate, num_epochs, num_folds)
	confi = np.reshape(confi, [1, len(confi)])[0]
	confi = np.array(confi, dtype = 'float64')
	n = len(confi)
	actual = np.array(actual)
	actual = actual[np.argsort(-confi)]
	confi = confi[np.argsort(-confi)]
	pos = 0
	neg = 0
	for i in range(len(actual)):
		if actual[i] == classes[-1][-1][1]:
			pos += 1
		else:
			neg += 1
	tp = 0
	fp = 0
	last_tp = 0
	for i in range(len(actual)):
		if i > 0 and confi[i] != confi[i-1] and actual[i] == classes[-1][-1][0] and tp > last_tp:
			fpr = float(fp)/float(neg)
			tpr = float(tp)/float(pos)
			res.append([fpr, tpr])
			last_tp = tp
		if actual[i] == classes[-1][-1][1]:
			tp += 1
		else:
			fp += 1
	fpr = float(fp)/float(neg)
	tpr = float(tp)/float(pos)
	res.append([fpr, tpr])
	x = []
	y = []
	for r in res:
		x.append(r[0])
		y.append(r[1])
	pylab.figure(3)
	pylab.plot(x, y, 'b', label = 'accuracy')
	pylab.title("ROC Curve")
	pylab.xlabel("False positive rate")
	pylab.ylabel("True positve rate")
	pylab.show()


class Network(object):
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.weight_init()


	def forward(self, input):
		data = input[:-1]
		for w, b in zip(self.weights, self.bias):
			data = sigmoid(np.dot(data, w) + b)[0]
		return data


	def weight_init(self):
		random.seed(10)
		self.bias = [(np.random.rand(1, n) - 0.5) * 2 for n in self.sizes[1:]]
		self.weights = [(np.random.rand(m, n) - 0.5)*2 for m, n in zip(self.sizes[:-1], self.sizes[1:])]


	def back_prop(self, x, y):
		db = [np.zeros(b.shape) for b in self.bias]
		dw = [np.zeros(w.shape) for w in self.weights]
		x = np.reshape(x, [1, len(x)])
		a = x
		activations = [x]
		zs = []
		for w, b in zip(self.weights, self.bias):
			z = np.dot(a, w) + b
			zs.append(z)
			a = sigmoid(z)
			activations.append(a)
		a = activations[-1]
		delta = a - y
		db[-1] = delta
		dw[-1] = np.dot(delta, activations[-2])
		for i in range(2, self.num_layers):
			z = zs[-i]
			prime = sigmoid_prime(z)
			delta = np.dot(self.weights[-i + 1], delta)
			delta = np.reshape(delta, [1, len(delta)])
			db[-i] = delta
			dw[-i] = np.dot(delta.transpose(), activations[-i - 1])
		return dw, db

	def update(self, data, eta):
		x = data[:-1]
		y = data[-1]			
		dw, db = self.back_prop(x, y)
		self.weights[0] = np.subtract(self.weights[0], float(eta)*dw[0].transpose())
		self.weights[1] = np.subtract(self.weights[1], float(eta)*dw[1].transpose())
		self.bias[0] = np.subtract(self.bias[0], float(eta)*db[0])
		self.bias[1] = np.subtract(self.bias[1], float(eta)*db[1])


	def sgd_cross(self, dataset, eta, epoch, num_folds):
		count = 0
		actual_class = []
		confidence = []
		n = len(dataset)		
		batch = split(dataset, num_folds)
		for i in range(num_folds):
			test = batch[i]
			training_data = batch[0:i] + batch[i+1:]
			for k in range(epoch):
				train_data = []			
				for batch_data in training_data:
					train_data += batch_data
				random.shuffle(train_data)
				for da in train_data:	
					self.update(da, eta)
			c, actual, confi = self.evaluate(test, i)
			count += c
			actual_class += actual
			confidence += confi
		return count/n, actual_class, confidence


	def evaluate(self, test_data, i):
		prediction = [self.forward(x) for x in test_data]
		pred = [int(p > 0.5) for p in prediction]
		temp = []
		actual_class = []
		confidence = []
		temp = np.array([int(t == int(y[-1])) for t, y in zip(pred, test_data)])
		accuracy = float(np.sum(temp))/float(len(prediction))
		for j in range(len(prediction)):
			print 'Fold : ', i + 1, 'predicted class :' , classes[-1][-1][int(pred[j])], \
			' Actual class : ', classes[-1][-1][int(test_data[j][-1])], \
			'confidence of prediction : ', prediction[j]
			actual_class.append(classes[-1][-1][int(test_data[j][-1])])
			confidence.append(prediction[j])
		# 	# print ("Fold '%d': predicted class:'%s', actual class:'%s', confidence of prediction: {:5.4f}".format(i, classes[-1][-1][int(temp[j])], classes[-1][-1][int(test_data[j][-1])], prediction[j]))
		return float(np.sum(temp)), actual_class, confidence


args = [arg for arg in sys.argv]
train = args[1]
num_folds = int(args[2])
learning_rate = float(args[3])
num_epochs = int(args[4])

train_data = sciarff.loadarff(train)
train = np.array([[i for i in train_data[0][j]] for j in range(train_data[0].shape[0])])

features = train_data[1].names()
classes = [train_data[1][feat] for feat in train_data[1].names()]
labels = classes[-1][1]

for i in range(len(train)):
	train[i][-1] = int(train[i][-1] == classes[-1][-1][-1])

train = np.array(train, dtype = 'float64')

net = Network([len(features)-1, len(features)-1, 1])

t, actual, confi = net.sgd_cross(train, learning_rate, num_epochs, num_folds)

epoches = [25, 50, 75, 100]
folds = [5, 10, 15, 20, 25]

# plot_roc(50, 0.1, 10)
# plot_folds(folds)
# plot_learning_curve(epoches)

