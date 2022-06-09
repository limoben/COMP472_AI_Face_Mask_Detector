import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn.functional as F
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset
import matplotlib.pyplot as plt

import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


#data loader
image_path = 'C:\study\concordia\comp472\project\COMP472_AI_Face_Mask_Detector\dataset'
#image_path = 'C:\study\concordia\comp472\project\COMP472_AI_Face_Mask_Detector\dataset_2class'
normalize = transforms.Normalize(
  mean=[0.485, 0.456, 0.406],
  std=[0.225, 0.225, 0.225]
)
transforms = transforms.Compose(
  [
  transforms.Resize([32, 32]), # resizing every image in 32*32 or 256*256
  transforms.ToTensor(), # transform to tensor
  normalize # ?????-->?????????,????????
  ]
)
data = torchvision.datasets.ImageFolder(image_path, transform=transforms)
print(data)
test_data_size = 1000
train_data_size = len(data) - test_data_size

# Splitting data into test and train
train_data, test_data = torch.utils.data.random_split(data, [train_data_size, test_data_size])
# def getTrainingData():
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=True)

# Evaluation attributes
y_train = np.array([y for x, y in iter(train_data)])
#classes = ('Cloth_Mask', 'N95_Mask', 'No_Mask', 'Surgical_Mask')
# Testing
# def displayImage():
#     images, labels = next(iter(train_loader))
#     print(images.shape, labels.shape)
#     for i in range(32):
#         plt.subplot(4, 8, i + 1)
#         plt.imshow(images[i].permute(1, 2, 0))
#     # plt.imshow(images[6].permute(1, 2, 0))
#     plt.show()
# displayImage()




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 *5 *5, 120) # 16 * 122 * 122 input size after conv and pooling layer(with 256*256 original)
        # important value : 16 *5 *5 is (original is 32*32)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4) # output final class 4

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 16 *5 *5)            
        x = F.relu(self.fc1(x))               
        x = F.relu(self.fc2(x))               
        x = self.fc3(x)                       
        return x



cnn = CNN()
print(cnn)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)   # learning rate=0.001


for epoch in range(5):
  # Training
  cnn.train()
  for i, (images, labels) in enumerate(train_loader):
    prediction = cnn(images)
    loss = loss_func(prediction, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total = labels.size(0)
    _, predicted = torch.max(prediction.data, 1)
    correct = (predicted == labels).sum().item()
    train_accuracy = correct/total

  # Testing
  cnn.eval()
  total = 0
  correct = 0
  for i, (images, labels) in enumerate(test_loader):
      prediction = cnn(images)

      total += labels.size(0)
      _, predicted = torch.max(prediction.data, 1)
      correct += (predicted == labels).sum().item()
      test_accuracy = correct/total
  print('Epoch: ' + str(epoch+1) + ' Train Loss: ' + str(loss.item()) + ' Train Accuracy: ' + str(train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))


torch.save(cnn.state_dict(), 'model.pth')

torch.manual_seed(0)

net = NeuralNetClassifier(
  CNN,
  max_epochs=1,
  iterator_train__num_workers=0,
  iterator_valid__num_workers=0,
  lr=1e-3,
  batch_size=64,
  optimizer=optim.Adam,
  criterion=nn.CrossEntropyLoss,
  device=torch.device("cpu")
)
net.fit(train_data, y=y_train)
y_pred = net.predict(test_data)
y_test = np.array([y for x, y in iter(test_data)])
accuracy_score(y_test, y_pred)
plot_confusion_matrix(net, test_data, y_test.reshape(-1, 1))
print(classification_report(y_test, y_pred))
plt.show()





























