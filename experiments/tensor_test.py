# import torch
# import numpy as np

# data = [[1,2], [3,4]]
# x_data = torch.tensor(data)

# print(x_data)

# x_ones = torch.ones_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")

# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")

# shape = (2, 3, )
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor} \n")

# tensor =  torch.rand(3, 4)

# if torch.cuda.is_available():
#   tensor = tensor.to("cuda")

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# tensor = torch.ones(4, 4)
# tensor[:,1] = 0
# print(tensor)

# t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# print(f"tensor * tensor \n {tensor * tensor}")

# print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)}")
# print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
# print(f"tensor @ tensor \n {tensor @ tensor}")

# print(tensor, "\n")
# tensor.add_(5)
# print(tensor)

# t = torch.ones(5)
# print(f"t: {t}")
# n = t.numpy()
# print(f"n: {n}")
# t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")

# n = np.ones(5)
# t = torch.from_numpy(n)
# np.add(n, 1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")

# from torchvision.models import resnet18, ResNet18_Weights
# model = resnet18(weights=ResNet18_Weights.DEFAULT)
# data = torch.rand(1, 3, 64, 64)
# labels = torch.rand(1, 1000)
# prediction = model(data) # forward pass
# loss = (prediction - labels).sum()
# loss.backward() # backward pass
# optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# optim.step() #gradient descent

# ===example 1
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16*5*5, 120) # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, input):
#         # Convolution layer C1: 1 input image channel, 6 output channels
#         # 5x5 square convolution, it uses RELU activation function, and
#         # outputs a Tensor with size (N, 6, 28, 28), where N is the size of batch
#         c1 = F.relu(self.conv1(input))
#         # Subsampling layer S2: 2x2 grid, purely functional
#         s2 = F.max_pool2d(c1, (2, 2))
#         # Convolution layer C3: 6 input channels, 16 output channels,
#         # 5x5 square convolution, it uses RELU activation function, and
#         # outputs a (N, 16, 10, 10) Tensor
#         c3 = F.relu(self.conv2(s2))
#         # Subsampling layer S4: 2x2 grid, purely functional,
#         # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
#         s4 = F.max_pool2d(c3, 2)
#         # Flatten operation: purely functional, outputs a (N, 400) Tensor
#         s4 = torch.flatten(s4, 1)
#         # Fully connected layer F5: (N, 400) Tensor input,
#         # and outputs a (N, 120) Tensor, it uses RELU activation function
#         f5 = F.relu(self.fc1(s4))
#         # fully connected layer F6: (N, 120) Tensor input,
#         # and outputs a (N, 84) Tensor, it uses RELU activation function
#         f6 = F.relu(self.fc2(f5))
#         # Fully connected layer OUTPUT: (N, 84) Tensor input, and
#         # outputs a (N, 10) Tensor
#         output = self.fc3(f6)
#         return output
    
# net = Net()
# print(net)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size()) # conv1's .weight

# input  = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)

# # net.zero_grad()
# # out.backward(torch.randn(1, 10))
# # print(out)

# output = net(input)
# target = torch.randn(10) # a dummy target
# target = target.view(1, -1) # make it the same shape as output
# criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss)

# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0]) # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

# net.zero_grad()
# print("conv1.bias.grid before backward")
# print(net.conv1.bias.grad)
# loss.backward()
# print("conv1.bias.grad after backward")
# print(net.conv1.bias.grad)

# import torch.optim as optim
# optimizer = optim.SGD(net.parameters(), lr=0.01)
# optimizer.zero_grad()
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()


#===2
# load and normalize the CIFAR10
import torch
import torchvision
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# show some trainning images
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# define a convolutional neural network
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# define a loss function and optimizer
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train the network
for epoch in range(2): # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pt'
torch.save(net.state_dict(), PATH)

# test the network on the test data
dataiter = iter(testloader)
images, labels = next(dataiter)
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# load saved model
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')