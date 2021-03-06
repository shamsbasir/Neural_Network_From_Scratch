import numpy as np 
import sys


class Sigmoid:
	def forward(self, x):
		self.state = 1 / (1 + np.exp(-x))
		return self.state

	def derivative(self):
		return self.state * (1 - self.state)


class SoftmaxCrossEntropy:
	def __init__(self):
		self.derivative_value = None

	def forward(self, x, y):
		batch_size = x.shape[0]
		#print("batch_size",batch_size)
		exp_prob = [x[i] - np.log(np.sum(np.exp(x[i]))) for i in range(batch_size)]
		self.derivative_value = -y + np.exp(exp_prob)
		loss = np.array([-np.sum(y[i] * exp_prob[i]) for i in range(batch_size)])
		return loss

	def derivative(self):
		return self.derivative_value

def uniform_weight_init(d0, d1):
	return np.random.uniform(-0.1, 0.1, (d0, d1))

def zeros_weight_init(d0, d1):
	return np.zeros((d0, d1))

def zeros_bias_init(d):
	return np.zeros(d)
	#return np.ones(d)

class NN:
	def __init__(self, input_size, hidden_size, output_size, lr, init_flag):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.activation = Sigmoid()
		self.output_size = output_size
		self.loss_function = SoftmaxCrossEntropy()
		self.lr = lr
		self.input = None
		self.output = None
		self.batch_size = None

		self.W = []
		self.b = []
		self.dW = []
		self.db = []

		if init_flag == 1:
			self.weight_init = uniform_weight_init
		else:
			self.weight_init = zeros_weight_init

		self.W.append(self.weight_init(input_size, hidden_size))
		self.W.append(self.weight_init(hidden_size, output_size))
		#alpha = np.array([[1,2,-3,0,1,-3],[3,1,2,1,0,2],[2,2,2,2,2,1],[1,0,2,1,-2,2]],dtype = float)
		#beta  = np.array([[1,2,-2,1],[1,-1,1,2],[3,1,-1,1]],dtype = float)
		#alpha = alpha.transpose()
		#beta  = beta.transpose()
		#self.W.append(alpha)
		#self.W.append(beta)
		self.b.append(zeros_bias_init(self.hidden_size))
		self.b.append(zeros_bias_init(self.output_size))
		#print("self.W = ",self.W)
		#print("self.b = ",self.b)
		for weights in self.W:
			#print("weights.shape",weights.shape)
			self.dW.append(np.zeros(weights.shape))
		for bias in self.b:
			self.db.append(np.zeros(bias.shape))

	def forward(self, x):
		self.input = x
		#print("self.input = ",x)
		self.batch_size = x.shape[0]
		#print("self.batch_size = ",self.batch_size)
		#print(" W.shape = ", self.W.shape)
		# LINEARFORWARD(X,a)
		a = np.dot(x, self.W[0]) + self.b[0]
		# SIGMOIDFORWARD(a)
		z = self.activation.forward(a)
		# LINEARFORWARD(Z,beta)
		b = np.dot(z, self.W[1]) + self.b[1]
		# SHOULD CALC CROSSENTROPYFORWARD(Y,YHAT)

		# RETURN THE OBJECT(x,a,z,b,yhat,J)
		self.output = b
		return self.output

	def backward(self, labels, test = False):
		loss = self.loss_function.forward(self.output, labels)
		if test:
			return loss
		delta_l = self.loss_function.derivative()
		for i in range(self.batch_size):
			self.dW[1] += np.dot(self.activation.state[i].reshape(self.hidden_size, 1), delta_l[i].reshape(1, -1))
			self.db[1] += delta_l[i]
		self.dW[1] /= self.batch_size
		self.db[1] /= self.batch_size
		delta_l = np.array([np.dot(self.W[1], delta_l[i]).flatten() * self.activation.derivative()[i].flatten()
							for i in range(self.batch_size)])
		for i in range(self.batch_size):
			self.dW[0] += np.dot(self.input[i].reshape(self.input_size, -1), delta_l[i].reshape(1, -1))
			self.db[0] += delta_l[i]
		self.dW[0] /= self.batch_size
		self.db[0] /= self.batch_size
		return loss

	def step(self):
		#print(" self.W",self.W)	
		for i in range(len(self.W)):
			self.W[i] -= self.lr * self.dW[i]
			self.b[i] -= self.lr * self.db[i]
		#print("self.W", self.W)
		#print("self.b",self.b)
		return

	def zero_grads(self):
		for i in range(len(self.W)):
			#print("i = dw " ,i)
			# initiate the gradients with zero
			self.dW[i] = np.zeros(self.W[i].shape)
			self.db[i] = np.zeros(self.b[i].shape)


# loading the data 
def load_data(path):
	labels = []
	data = []
	with open(path, "r") as f:
		lines = f.readlines()
	for line in lines:
		label_vec = np.zeros(10)
		label_vec[int(line.split(',')[0])] = 1
		labels.append(label_vec.reshape(1, 10))
		data_vec = [int(item) for item in line.split(',')[1:]]
		data.append(np.array(data_vec).reshape(1, 128))
	return data, labels

# run 
def run(train_input, test_input, train_out, test_out, metrics_out, num_epoch, hidden_units, init_flag, lr):
	train_data, train_labels = load_data(train_input)
	test_data, test_labels = load_data(test_input)

	model = NN(input_size = 128, hidden_size = hidden_units, output_size = 10, lr = lr, init_flag = init_flag)
	metrics_out_file = open(metrics_out, "w")
	train_out_file = open(train_out, "w")
	test_out_file = open(test_out, "w")
	for epoch in range(num_epoch):
		for feat, label in zip(train_data, train_labels):
			model.zero_grads()
			out = model.forward(feat)
			loss = model.backward(label)
			#print("loss =",loss)
			model.step()
		train_loss = 0.0
		for feat, label in zip(train_data, train_labels):
			model.zero_grads()
			out = model.forward(feat)
			loss = model.backward(label)
			train_loss += np.sum(loss)
		train_loss /= len(train_data)
		#print("len(train_data)",len(train_data))
		metrics_out_file.write("epoch={} crossentropy(train): {}\n".format(epoch + 1, train_loss))

		test_loss = 0.0
		for feat, label in zip(test_data, test_labels):
			model.zero_grads()
			out = model.forward(feat)
			loss = model.backward(label)
			test_loss += np.sum(loss)
		test_loss /= len(test_data)
		metrics_out_file.write("epoch={} crossentropy(test): {}\n".format(epoch + 1, test_loss))

	error = 0
	for feat, label in zip(train_data, train_labels):
		model.zero_grads()
		out = model.forward(feat)
		pred = np.argmax(out)
		train_out_file.write("{}\n".format(pred))
		error += (np.argmax(label) != pred)
	metrics_out_file.write("error(train): {}\n".format(error / len(train_data)))

	error = 0
	for feat, label in zip(test_data, test_labels):
		model.zero_grads()
		out = model.forward(feat)
		pred = np.argmax(out)
		test_out_file.write("{}\n".format(pred))
		error += (np.argmax(label) != pred)
	metrics_out_file.write("error(test): {}\n".format(error / len(test_data)))



if __name__ == "__main__":
	_, train_input, test_input, train_out, test_out, metrics_out, num_epoch, hidden_units, init_flag, lr = sys.argv
	run(train_input, test_input, train_out, test_out, metrics_out, int(num_epoch), int(hidden_units), int(init_flag), float(lr))


