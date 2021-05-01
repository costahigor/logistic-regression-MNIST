import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

class MNIST_LR(nn.Module):
	def __init__(self):
		super().__init__()
		self.lin = nn.Linear(784, 10)

	def forward(self, x):
		return self.lin(x)

# Load data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)


# TRAINING

# Instantiate model
model = MNIST_LR()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Iterate train set minibatchs
for images, labels in tqdm(train_loader):
	# Zero out the gradients
	optimizer.zero_grad()

	# Forward pass
	x = images.view(-1, 28*28)
	y = model(x)
	loss = criterion(y, labels)
	# Backward pass
	loss.backward()
	optimizer.step()


# TESTING
correct = 0
total = len(mnist_test)

with torch.no_grad():
	# Iterate test set minibatchs
	for images, labels in tqdm(test_loader):
		# Forward pass
		x = images.view(-1, 28*28)
		y = model(x)

		predictions = torch.argmax(y, dim=1)
		correct += torch.sum((predictions == labels).float())

print("\nTest accuracy: {}".format(correct/total))
