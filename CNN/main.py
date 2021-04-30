
#%%
import os
from model import LeNet5
from model import CustomMLP
from dataset import MNIST
from torchvision import transforms
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#%%

# import some packages you need here
os.chdir('./template')

def train(model, train_loader, DEVICE, optimizer, epoch):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    
    model.train()
    
    trn_loss=0
    correct = 0
        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output=model(data)
        
        loss = F.cross_entropy(output, target)
        
        trn_loss += F.cross_entropy(output, target, reduction='sum').item()
        
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
               
        loss.backward()
        optimizer.step()
              
    
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,batch_idx * len(data), len(train_loader.dataset), 
                  100. * batch_idx/len(train_loader), loss.item()))
    
    trn_loss /=len(train_loader.dataset)
    tr_acc = 100. * correct / len(train_loader.dataset)
    print('train_loss: {:.4f}, tr_acc: {:.2f}%'.format(trn_loss, tr_acc))
    
    return trn_loss, tr_acc



#%%

def test(model, tst_loader, DEVICE):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()
    tst_loss = 0
    correct = 0
    
    with torch.no_grad() :
        for (data, target) in tst_loader :
            
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            tst_loss+= F.cross_entropy(output, target, reduction='sum').item()
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        tst_loss /= len(tst_loader.dataset)
        acc = 100. * correct / len(tst_loader.dataset)
    return tst_loss, acc

#%%       

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    
    EPOCHS = 10
    BATCH_SIZE = 64
    
    model  = LeNet5().to(DEVICE)
    #model = CustomMLP().to(DEVICE)
    #optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.5, weight_decay=0.001)
    
    
    d =  "C:/Users/inolab/Desktop/DNN/CNN_homework/data"
    
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    transform_test = transforms.Compose([transforms.ToTensor()])
    
    train_set = MNIST(data_dir=d, folder='train', transform = transform_train)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    test_set = MNIST(data_dir=d, folder='test', transform = transform_test)
    tst_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    for epoch in range(1, EPOCHS + 1):
        trn_loss, tr_acc = train(model, train_loader,DEVICE, optimizer, epoch)
        tst_loss, acc= test(model, tst_loader, DEVICE)
        
        print('[{}] Test Loss : {:.4f}, Test Acc: {:.2f}%'.format(epoch, tst_loss, acc))
        
#%%    
        
if __name__ == '__main__':
    main()
    

