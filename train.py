import torch
import torchvision
from torchvision import transforms
from models import *
import os

out_dir = 'out'
model_name = 'model.pth'

# Define transformations for the data
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,)) 
])

# Load the training and testing datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = CNN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, )

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'[epoch {epoch + 1}, batch {i + 1}] loss: {running_loss}')
        running_loss = 0.0

print('Finished Training')
os.makedirs(out_dir, exist_ok=True)
model_path = os.path.join(out_dir, model_name)
torch.save(model.state_dict(), model_path)