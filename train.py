from  src.CNN import CNN
from  src.dataset import  get_data_loaders 
import torch
import torch.nn as nn


device = "cpu"
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

train_loader,test_loader,train_dataset,test_dataset = get_data_loaders()

epoches = 3
running_loss = 0.0
for epoch in range(epoches):
    running_loss = 0.0
    model.train()
    for images , labels in train_loader:
        images , labels = images.to(device),labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"epoch {epoch+1}/{epoches} completed | average loss : {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
with torch.no_grad():
    for data , target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
torch.save(model.state_dict(),'./models/mnist_cnn.pth')
print(f"\n Final test accuracy :{100*correct  /len(test_dataset):.2f}")