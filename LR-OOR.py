import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

class xW_plus_b:
	def __init__(self, dim_in, dim_out):
		self.W = torch.randn(dim_in, dim_out)/np.sqrt(dim_in)
		self.W.requires_grad_()
		self.b = torch.zeros(dim_out, requires_grad=True)

	def forward(self, x):
		return torch.matmul(x, self.W) + self.b

# Load data
mnist_train  = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test   = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader  = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

# TRAINING

# Initialize Parameters
lin = xW_plus_b(784, 10)

# Optmizer
optimizer = torch.optim.SGD([lin.W, lin.b], lr=0.1)

# Interate through train set minibatchs
for images, labels in tqdm(train_loader):
	# Zero out the gradients
	optimizer.zero_grad()

	# Forward pass
	x = images.view(-1, 28*28)
	y = lin.forward(x)
	cross_entropy = F.cross_entropy(y, labels)

	# Backward pass
	cross_entropy.backward()
	optimizer.step()


# TESTING
correct = 0
total = len(mnist_test)

with torch.no_grad():
	# Interate through test set minibatchs
	for images, labels in tqdm(test_loader):
		# Forward pass
		x = images.view(-1, 28*28)
		y = lin.forward(x)

		predictions = torch.argmax(y, dim=1)
		correct += torch.sum((predictions == labels).float())

	print("Test accuracy: {}".format(correct/total))
