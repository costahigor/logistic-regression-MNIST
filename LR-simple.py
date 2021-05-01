import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

# Load data
mnist_train  = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test   = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader  = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

print("\nNumber of MNIST training examples: {}".format(len(mnist_train)))
print("Number of MNIST test examples: {}\n".format(len(mnist_test)))

# Visualizing a item from the training set
image, label = mnist_train[7]
# Plot the image
print("Default image shape: {}".format(image.shape))
image = image.reshape([28, 28])
print("Reshaped image shape: {}".format(image.shape))
plt.imshow(image, cmap="gray")
print("Image label: {}\n".format(label))

# TRAINING

# Initialize Parameters
W = torch.randn(784, 10)/np.sqrt(784)
W.requires_grad_()
b = torch.zeros(10, requires_grad=True)

# Optmizer
optimizer = torch.optim.SGD([W, b], lr=0.1)

# Interate through train set minibatchs
for images, labels in tqdm(train_loader):
	# Zero out the gradients
	optimizer.zero_grad()

	# Forward pass
	x = images.view(-1, 28*28)
	y = torch.matmul(x, W) + b
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
		y = torch.matmul(x, W) + b

		predictions = torch.argmax(y, dim=1)
		correct += torch.sum((predictions == labels).float())

	print("Test accuracy: {}".format(correct/total))


# Visualizing the weights
fig, ax = plt.subplots(1, 10, figsize=(20, 2))
for digit in range(10):
	ax[digit].imshow(W[:, digit].detach().view(28,28), cmap="gray")
