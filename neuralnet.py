import numpy as np
import sys
class NN:
	def __init__(self):
		self.alpha = None
		self.beta  = None
		return None
	
	def SGS(self):
		return None

	def Predict(self,infile,outfile):
		return 1

	def NNForward(self):
		return None

	def NNBackword(self):
		return None
	
	def CrossEntropy(self):
		return None
	
	def FiniteDiff(self):
		return None

	def Train(self,infile,epoch,hUnit,iflag,lr):
		data, labels = self.load(infile)
		print(" labels[0] =  ",labels[0])
		print(" data[0]   = ",data[0])
		return None
	
	def load(self, infile):
		labels = []
		data   = []
		with open(infile, "r") as file:
			text = file.readlines()
		for row in text:
			# there are 10 labels
			label = np.zeros(10)
			# make them one-hot encoding
			label[int(row.split(',')[0])] = 1
			# grab the labels
			labels.append(label.reshape(1, 10))
			# grab the features
			feature = [float(item) for item in row.split(',')[1:]]
			# print("feature", feature[4])
			# append the features into a numpy array
			data.append(np.array(feature).reshape(1, np.size(feature)))
		return data, labels

	
def postProcess(trError,tstError,numEpoch,metrics):
	return None

def main():
	trainIfile = sys.argv[1]
	testIfile  = sys.argv[2]
	trainOfile = sys.argv[3]
	testOfile  = sys.argv[4]
	metrics    = sys.argv[5]
	numEpoch   = int(sys.argv[6])
	numHidUnit = int(sys.argv[7])
	initFlag   = int(sys.argv[8])
	learnRate  = float(sys.argv[9])
	# Creating our one-Hidden Layer Nueral Network	
	NeuralNet = NN()
	# training our NN
	NeuralNet.Train(trainIfile,numEpoch,numHidUnit,initFlag,learnRate)
	# predicting the training data 
	trError   = NeuralNet.Predict(trainIfile,trainOfile)
	# predicting the test data
	tstError  = NeuralNet.Predict(testIfile,testOfile)
	# outputing the test data 
	postProcess(trError,tstError,numEpoch,metrics)
	
	
if __name__=="__main__":
	main()
 
