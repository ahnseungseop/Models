
# import some packages you need here
import torch
import os
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob
import warnings
warnings.filterwarnings('ignore')

#%%
class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, folder, transform=None):
        
        self.img_list = sorted(glob(os.path.join(data_dir, folder, '*')))
               
        self.transform = transform    
        
                

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx):

        # write your codes here
        
        data_path = self.img_list[idx]
        img = io.imread(data_path)
        
        if self.transform:
            img = self.transform(img) 
            
        img_name = os.path.basename(data_path)
        for i in range(0,10) :
            if i == int(img_name[6]):
                label = torch.tensor(i)
                break
            else :
                continue
            
        
        return  img, label
    
#%%
        
if __name__ == '__main__':

    # write test codes to verify your implementations
    
    d = "./data"
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    BATCH_SIZE=10
    
    
    train_set = MNIST(data_dir=d, folder='train', transform = transform_train)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    for i in range(0,4) :
        a = train_loader
        print(a)

