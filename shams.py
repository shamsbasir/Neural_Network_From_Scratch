import numpy as np 
import sys

class SigmoidModule:
	def SigmoidForward(self,a):
		z = 1/(1+np.exp(-a)) 
		return z
	def SigmoidBackward(self,a,b,g_b):
		return None

class LinearModule:
	def LinearForward(self,x,alpha):
		a = np.dot(alpha,x)
		return 
	def LinearBackward(self):
		return None

class SoftmaxModule:
	def softmaxForward(self):
		return None
	def softmaxBackward(self):
		return None

class CrossEntropyModule:
	def CrossEntropyForward(self):
		return None
	def CrossEntropyBackward(self):
		return None

class NN:
	def __init__(self):
		return None
	def NNForward(self):
		return None
	def NNBackward(self):
		return None
	def SGS(self):
		return None
	def Predict(self):
		return None


if __name__ == "__main__":
#	_, train_input,num_epoch, hidden_units, init_flag, lr = sys.argv
	
