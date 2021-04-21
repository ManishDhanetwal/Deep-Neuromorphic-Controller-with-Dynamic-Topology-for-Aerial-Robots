import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import pdb
from termcolor import colored, cprint
from tools import *

# hidden layer class definition
class hiddenlayer(nn.Module):
	def __init__(self, no_input, no_output):
		super(hiddenlayer, self).__init__()
		self.linear = nn.Linear(no_input, no_output,bias=False)
		nn.init.xavier_normal_(self.linear.weight,gain=0.01)
		
		if self.linear.bias is not None:
			nn.init.zeros_(self.linear.bias)
			#nn.init.xavier_uniform_(self.linear.bias)
		self.layerCorr = recCorr()
		self.adaptLr  = 0.01
		self.out = 0.0
		#self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

	def forward(self, x):
		x = self.linear(x)
		#x = self.bn(x)
		self.out = F.tanh(x)        
		return self.out

# Create hidden layer function
def createhiddenlayer(no_input,no_output):
	obj = hiddenlayer(no_input,no_output).double()
	return obj

# Output layer class definition
class outputlayer(nn.Module):
	def __init__(self, no_input, classes):
		super(outputlayer, self).__init__()
		self.linear = nn.Linear(no_input, classes,bias=False)
		#nn.init.xavier_normal_(self.linear.weight,gain=0.01)
		#nn.init.zeros_(self.linear.weight)
		
		if self.linear.bias is not None:
			nn.init.zeros_(self.linear.bias)
			#nn.init.xavier_uniform_(self.linear.bias)
		#self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

	def forward(self, x):
		x = self.linear(x)
		#x = F.tanh(x) 
		return x

# Create output layer function
def createoutputlayer(no_input,classes):
	obj = outputlayer(no_input,classes).double()
	return obj

# Evolving NNSMC Class Controller definition
class NetEvo(object):
	def __init__(self,nInput,nOutput,initNode):
		self.netStruct = [createhiddenlayer(nInput,initNode),createoutputlayer(initNode,nOutput)]
		self.netBias   = recursiveMeanVar()
		self.netVar    = recursiveMeanVar()
		self.netmVar   = recursiveMeanVar()
		self.evalBias  = recursiveMeanVar()

		self.cnt     = 0
		self.err     = 0.0
		self.dt      = 0.1
		self.smcpar  = [0.0,0.0,0.0]
		self.us_last = 0.0
		self.us      = 0.0
		self.us_dot  = 0.0
		self.un      = 0.0
		self.un_out  = 0.0
		self.err     = 0.0
		self.derr    = 0.0
		self.interr  = 0.0
		self.lasterr = 0.0
		self.last2err= 0.0
		self.filteredderr = 0.0
		self.windup  = 0.0
		self.uNN     = 0.0
		self.u       = 0.0
		self.nNodes  = initNode
		self.nLayer  = len(self.netStruct)
		self.g       = 0.0
		self.mBias	 = 0.0
		self.varBias = 0.0
		self.Var     = 0.0
		self.mVar    = 0.0
		self.stdBias = 0.0
		self.stdVar  = 0.0
		self.momBias = 0.0   
		self.momBias_old = 0.0
		self.minBias     = 0.0
		self.minStdBias  = 0.0
		self.minVar      = 0.0
		self.minStdVar   = 0.0
		self.gradBias    = 0.0
		self.justGrow  = False
		self.justPrune = False


	def calcBiasVar(self,forgetFactor,nWindow):
		# Mean Bias and Variance 
		sysBias                 = 0.5*((self.us_dot+self.us))**2					# Bias of u Equivalent
		self.mBias,self.varBias = self.netBias.updateMean(sysBias,forgetFactor,nWindow)
		self.stdBias            = np.sqrt(self.varBias)

		self.mUn,self.Var       = self.netVar.updateMean(self.un,forgetFactor,nWindow)
		self.mVar,self.varVar   = self.netmVar.updateMean(self.Var,forgetFactor,nWindow)
		self.stdVar             = np.sqrt(self.varVar)

	def calcMomBiasVar(self,forgetFactor,nWindow):
		# Mean of Mean Bias and Variance
		self.momBias,varMomBias = self.evalBias.updateMean(self.mBias,forgetFactor,nWindow/2)
		self.gradBias           = (self.momBias-self.momBias_old)/(nWindow*self.dt)
		self.momBias_old        = self.momBias
		self.tBias              = self.cnt*self.dt

	def forward(self,x):
		# Feedforward all layers
		for i in range(len(self.netStruct)):
			x = self.netStruct[i](x)

		# Calculate Hidden nodes correlation as learning rate factor
		nLayers=len(self.netStruct)
		for i in range(nLayers-1):
			nNodes   = len(self.netStruct[i].out)
			nodeCorr = torch.empty(nNodes)

			for j in range(nNodes):
				nodeCorr[j]=abs(self.netStruct[i].layerCorr.updateCorr(self.netStruct[i].out[j],x,0.98,35))

			self.netStruct[i].adaptLr = nodeCorr.mean()
		return x

	def addhiddenlayer(self):
		# Add layer by inserting new layer at the top of hidden layer or right before output layer
		n_node  = self.netStruct[len(self.netStruct)-2].linear.out_features
		addedLayer = createhiddenlayer(n_node,n_node)
		addedW     = torch.ones(n_node, n_node,dtype=torch.double)
		#addedW     = 0.01*torch.rand(n_node, n_node,dtype=torch.double)*np.sqrt(1/2*n_node)

		addedLayer.linear.weight.data = addedW.clone().detach().requires_grad_(True)
		newNet  = np.copy(np.insert(self.netStruct, len(self.netStruct)-1, addedLayer))

		self.netStruct = newNet
		self.nLayer = len(self.netStruct)
		for i in range(self.nLayer):
			self.optimizer = torch.optim.Adam(self.netStruct[i].parameters(), lr=0.05)

		text = colored('A new layer is created, total layer = %s'%len(self.netStruct), 'yellow', attrs=['reverse', 'blink'])
		print (text )

	def growNode(self):
		# Growing the node only applied for the latest hidden layer (closest to output)
		nLayer = len(self.netStruct)
		adjustedLayer = nLayer - 2
		outputLayer   = nLayer - 1

		nInputAdjusted = self.netStruct[adjustedLayer].linear.in_features
		nOutput        = self.netStruct[outputLayer].linear.out_features

		W     = copy.deepcopy(self.netStruct[adjustedLayer].linear.weight.data)
		W_out = copy.deepcopy(self.netStruct[outputLayer].linear.weight.data)

		W_add     = 0.01*torch.rand(1, nInputAdjusted,dtype=torch.double)*np.sqrt(1/(nInputAdjusted+nOutput))
		W_out_add = 0.01*torch.rand(nOutput, 1,dtype=torch.double)*np.sqrt(1/(nInputAdjusted+nOutput))

		W     = torch.cat((W,W_add), 0)
		W_out = torch.cat((W_out,W_out_add), 1)

		self.netStruct[adjustedLayer].linear.out_features += 1
		self.netStruct[outputLayer].linear.in_features    += 1

		L1 = []
		L2 = []
		n_inNew  = self.netStruct[adjustedLayer].linear.in_features
		n_outNew = self.netStruct[adjustedLayer].linear.out_features

		L1 = createhiddenlayer(n_inNew,n_outNew)
		L2 = createoutputlayer(n_outNew,nOutput)
		L1.linear.weight.data = copy.deepcopy(W)
		L2.linear.weight.data = copy.deepcopy(W_out)

		if self.netStruct[adjustedLayer].linear.bias is not None:
			b        = copy.deepcopy(self.netStruct[adjustedLayer].linear.bias.data)
			bias_add = torch.zeros(1,dtype=torch.double)
			b        = torch.cat((b,bias_add), 0)
			L1.linear.bias.data = copy.deepcopy(b)
			L2.linear.bias.data = copy.deepcopy(self.netStruct[outputLayer].linear.bias.data)

		self.netStruct[adjustedLayer] = copy.deepcopy(L1)
		self.netStruct[outputLayer]   = copy.deepcopy(L2)

		text = colored('A node is added..!!', 'blue', attrs=['reverse', 'blink'])
		print (text)

	def pruneNode(self):
		# Pruning node mechanism only applied for the latest hidden layer (closest to output)
		nLayer = len(self.netStruct)
		prunedLayer = nLayer - 2
		outputLayer = nLayer - 1

		W = copy.deepcopy(self.netStruct[prunedLayer].linear.weight.data)
		W_out = copy.deepcopy(self.netStruct[outputLayer].linear.weight.data)
		nClass = self.netStruct[outputLayer].linear.out_features

		minNode = torch.argmin(torch.sum(W**2,axis=1))

		W = np.delete(W,minNode, 0)
		W_out = np.delete(W_out,minNode, 1)

		self.netStruct[prunedLayer].linear.out_features -= 1
		self.netStruct[outputLayer].linear.in_features  -= 1

		L1 = []
		L2 = []
		n_inNew  = self.netStruct[prunedLayer].linear.in_features
		n_outNew = self.netStruct[prunedLayer].linear.out_features
		L1 = createhiddenlayer(n_inNew,n_outNew)
		L2 = createoutputlayer(n_outNew,nClass)
		L1.linear.weight.data = copy.deepcopy(W)
		L2.linear.weight.data = copy.deepcopy(W_out)
		
		if self.netStruct[prunedLayer].linear.bias is not None:
			b = copy.deepcopy(self.netStruct[prunedLayer].linear.bias.data)
			b = np.delete(b,minNode, 0)
			L1.linear.bias.data   = copy.deepcopy(b)
			L2.linear.bias.data   = copy.deepcopy(self.netStruct[outputLayer].linear.bias.data)

		self.netStruct[prunedLayer] = copy.deepcopy(L1)
		self.netStruct[outputLayer] = copy.deepcopy(L2)

		text = colored('Node : %s is pruned..!!'%minNode, 'red', attrs=['reverse', 'blink'])
		print (text)

	def adjustWidth(self,FORGET_FACTOR,N_WINDOW):
		# Number of node adjustment mechanism
		self.justGrow  = False
		self.justPrune = False

		self.calcBiasVar(FORGET_FACTOR,N_WINDOW)

		if self.cnt % N_WINDOW==0:

			self.calcMomBiasVar(FORGET_FACTOR,N_WINDOW)

			if self.cnt==N_WINDOW or self.cnt==0:
				self.minBias    = self.mBias
				self.minStdBias = self.stdBias
				self.minVar     = self.mVar
				self.minStdVar  = self.stdVar

			# Grow Node Mechanism
			Xi = (1.3*np.exp(-1*(self.mBias))+0.7)

			if (self.mBias+self.stdBias) > (self.minBias+Xi*self.minStdBias) and not self.justPrune:
				self.growNode()
				self.nNodes   += 1
				self.justGrow  = True

				self.minBias    = self.mBias
				self.minStdBias = self.stdBias

			# Prune Node Mechanism
			if self.cnt !=0 and self.nNodes > 1 and not self.justGrow :

				phii  = 2*(1.3*np.exp(-1*(self.mVar))+0.7)

				if (self.mVar+self.stdVar) > (self.minVar+phii*self.stdVar):
					self.pruneNode()
					self.nNodes -= 1
					self.justPrune  = True

					self.minBias    = self.mBias
					self.minStdBias = self.stdBias				

	def adjustDepth(self,ETA,DELTA,N_WINDOW,EVAL_WINDOW):
		# Only add a new Network Layer
		if self.cnt % (EVAL_WINDOW*N_WINDOW) == 0 and self.cnt!=0:		
			#pdb.set_trace()
			if self.cnt == (EVAL_WINDOW*N_WINDOW):       
				self.minGradBias = self.gradBias
				self.maxGradBias = 0.0

			if self.maxGradBias < self.gradBias:
				self.maxGradBias = self.gradBias

			gradRange = self.maxGradBias - self.minGradBias

			if self.mVar <= ETA:
				self.mVar = ETA

			right = gradRange/self.mVar*np.sqrt(1/2*EVAL_WINDOW*np.log(1/DELTA))

			if self.gradBias > 0 and abs(self.gradBias - self.minGradBias) > right and self.nNodes>=2:
				#pdb.set_trace()
				self.addhiddenlayer()
				self.nLayer =len(self.netStruct)
				self.minGradBias = self.gradBias

	def calculateError(self,yr,yout):
		# Trajectory error calculations
		self.last2err     = self.lasterr
		self.lasterr      = self.err
		self.err          = yr-yout
		self.derr         = (self.err-self.lasterr)/self.dt
		self.filteredderr = self.filteredderr+((-1*self.filteredderr*0.5)+self.derr)*self.dt
		self.interr       = self.interr+((self.err+0.015*self.windup)*self.dt)
		self.cnt         += 1

		return self.err,self.derr,self.interr

	def controlUpdate(self,yr):
		# Updating the control signal for both SMC and NN
		Xe = np.array([self.err,self.lasterr,self.last2err,yr])
		Xe = torch.from_numpy(Xe.T)
		Xe = torch.tensor(Xe,dtype=torch.double)
		Xe.requires_grad_(True)
		
		self.uNN     = self.forward(Xe)
		self.un      = np.asscalar(self.uNN.data.numpy())
		#self.un      = limiter(self.un,0.8)
		self.us      = self.smcpar[0] * self.err + self.smcpar[1] * self.interr + self.smcpar[2] * self.derr
		#self.us      = limiter(self.us,0.4)
		self.us_dot  = (self.us - self.us_last)/self.dt
		self.us_dot  = limiter(self.us_dot,0.01)
		self.u       = self.us + 1.0*self.un
	
		self.us_last = self.us
	
		return self.u

	def optimize(self,lr,alpha):
		# Optimize the neural network with adaptive learning rate

		loss=0.5*(self.u-self.uNN)**4 + 0.5*(self.err)**2 #+ 0.5*self.err**2
		#loss = torch.tensor([0.5*self.err**2],dtype=torch.double,requires_grad_=True)

		loss.backward()							# Backward the NN output
		
		nLayers=len(self.netStruct)

		for i in range(nLayers):
			if i < nLayers-1:
				for param_group in self.netStruct[i].optimizer.param_groups:
					param_group['lr'] = np.asscalar(self.netStruct[i].adaptLr.data.numpy())

			self.netStruct[i].optimizer.step()
			self.netStruct[i].optimizer.zero_grad()
	
		# nLayers=len(self.netStruct)
		# for i in range(nLayers):
		# 	#pdb.set_trace()
		# 	if i < nLayers-1:
		# 		self.g = lr *self.netStruct[i].adaptLr * (self.us_dot+self.us) # Adaptive learning rate and SMC derivative update law
		# 	else:
		# 		self.g = lr * (self.us_dot+self.us)

		# 	W = self.netStruct[i].linear.weight.data + torch.tensor(self.g,dtype=torch.double)*self.netStruct[i].linear.weight.grad + alpha*self.netStruct[i].linear.weight.data
		# 	self.netStruct[i].linear.weight.data  = W.clone().detach().requires_grad_(True)

		# 	if self.netStruct[i].linear.bias is not None:
		# 		b = self.netStruct[i].linear.bias.data + torch.tensor(self.g,dtype=torch.double)*self.netStruct[i].linear.bias.grad + alpha*self.netStruct[i].linear.bias.data
		# 		self.netStruct[i].linear.bias.data  = b.clone().detach().requires_grad_(True)

		# 	if i==0:
		# 		#pdb.set_trace()
		# 		print 'grad    = ', self.netStruct[0].linear.weight.grad
		# 		print 'adaptLr = ', self.netStruct[i].adaptLr

		# 	# Reset weight Gradients
		# 	self.netStruct[i].linear.weight.grad.zero_()

	# def optimizeAdam(self,lr,beta1=torch.tensor([0.9],dtype=torch.double),beta2=torch.tensor([0.999],\
	# 	dtype=torch.double),eta=torch.tensor([10**(-8)],dtype=torch.double)):
		
	# 	self.uNN.backward(torch.ones_like(self.uNN))
	# 	self.g = lr * (self.us_dot+self.us)

	# 	for i in range(len(self.netStruct)-1):
	# 		#pdb.set_trace()
	# 		self.momentumW[i] = beta1*self.momentumW[i]+(1-beta1)*self.netStruct[i].linear.weight.grad.data
	# 		self.rateW[i]     = beta2*self.rateW[i]+(1-beta1)*(self.netStruct[i].linear.weight.grad.data)**2
	# 		estMomentumW = self.momentumW[i]/(1-(beta1**self.cnt))
	# 		estRateW     = self.rateW[i]/(1-(beta2**self.cnt))

	# 		self.momentumB[i] = beta1*self.momentumB[i]+(1-beta1)*self.netStruct[i].linear.bias.grad.data
	# 		self.rateB[i]     = beta2*self.rateB[i]+(1-beta1)*(self.netStruct[i].linear.bias.grad.data)**2
	# 		estMomentumB = self.momentumB[i]/(1-(beta1**self.cnt))
	# 		estRateB     = self.rateB[i]/(1-(beta2**self.cnt))

	# 		adamWGrad = estMomentumW/np.sqrt(estRateW+eta)
	# 		adamBGrad = estMomentumB/np.sqrt(estRateB+eta)

	# 		W = self.netStruct[i].linear.weight.data + torch.tensor(self.g,dtype=torch.double)*adamWGrad + 0.000125*self.netStruct[i].linear.weight.data
	# 		b = self.netStruct[i].linear.bias.data + torch.tensor(self.g,dtype=torch.double)*adamBGrad + 0.000125*self.netStruct[i].linear.bias.data
			
	# 		self.netStruct[i].linear.weight.data  = W.clone().detach().requires_grad_(True)
	# 		self.netStruct[i].linear.bias.data    = b.clone().detach().requires_grad_(True)


#####################
#### EXPERIMENTS ####------------------------------------------------------------------------
#####################

# N_INPUT  = 3
# N_OUTPUT = 1
# INIT_NODE = 5
# LEARN_RATE = 0.0001
# WEIGHT_DECAY  = 0.000125 

# netEv = NetEvo(N_INPUT,N_OUTPUT,INIT_NODE)

# yout = 0.5
# yr   = 1

# err,derr,interr = netEv.calculateError(yr,yout)
# print netEv.netStruct
# netEv.addhiddenlayer()
# pdb.set_trace()
# print netEv.netStruct
# u = netEv.controlUpdate()

# print u



# #pdb.set_trace()

# netEv.optimize(LEARN_RATE,WEIGHT_DECAY)
# netEv.growNode()

# pdb.set_trace()