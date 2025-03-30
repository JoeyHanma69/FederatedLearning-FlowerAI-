import torch 
import torch.nn as nn  
import torch.nn.functional as F 
from torchvision.transforms import Compose, ToTensor, Normalize 
from torch.utils.data import DataLoader 
from torchvision.datasets import CIFAR10  


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):   
    """Model (simple CNN adapted from PyTorch: A 60 Minute Blitz)"""
    def __init__(self): 
        super(Net, self).__init__() 
        self.cov1 = nn.Conv2d(3, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 
        
    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 16 * 5 * 5) 
        x = F.view(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        return self.fc3(x)  
    
    
    
def train(net, trainloader, epochs, device): 
    """Train the model on the training set""" 
    net.to(DEVICE) # move model to GPU if available 
    criterion = torch.nn.CrossEntropyLoss().to(device) 
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
    for _ in range(epochs):  
        for images, labels in trainloader: 
            optimizer.zero_grad() 
            criterion(net(images.to(device)), labels.to(device)).backward()  
            optimizer.step() 
                         
    
def test(net, testloader, device): 
    """Validate the model on the test set. """
    net.to(device) 
    criterion = torch.nn.CrossEntropyLoss() 
    correct, loss = 0, 0.0 
    with torch.no_grad():  
        for images, labels in testloader: 
            outputs = net(images.to(DEVICE)) 
            loss += criterion(outputs, labels.to(DEVICE)).item()  
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item() 
    return loss / len(testloader.dataset), correct / total 
  
def load_data():
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
    
    
def load_model():
    return Net().to(DEVICE)  
         
if __name__ == "__main__": 
    net = load_model() 
    trainloader, testloader = load_data() 
    train(net, trainloader, 5) 
    loss, accuracy = test(net, testloader) 
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
    

        
        
    