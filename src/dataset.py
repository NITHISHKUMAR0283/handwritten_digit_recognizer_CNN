from torch.utils.data import DataLoader
from torchvision import datasets,transforms

def get_data_loaders():
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(
        root = "./data/train_data",
        train=True,
        transform = transform,
        download = True
    )
    test_dataset =  datasets.MNIST(
        root = "./data/test_data",
        train=False,
        transform = transform,
        download =True
    )

    train_loader = DataLoader(train_dataset,batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset,batch_size = 64,shuffle = True)

    return train_loader,test_loader,train_dataset,test_dataset