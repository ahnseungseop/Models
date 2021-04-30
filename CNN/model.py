#%%

import torch.nn as nn
import torch.nn.functional as F

#%%

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):

        # write your codes here
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 흑백 이미지 색상채널 1, 10개의 feature map
        self.conv2 = nn.Conv2d(10,20, kernel_size=5) # 10개의 feature map을 받아 20개의 feature map 생성
        
        #self.drop = nn.Dropout2d() # nn모듈을 이용한 dropout
        
        self.fc1 = nn.Linear(320, 50) # 앞 계층의 출력크기인 4*4*20 을 입력크기로 함
        self.fc2 = nn.Linear(50, 10) # 두번째 계층에선 입력 크기를 50, 출력 크기 10 = 클래스
                
    def forward(self, x):
        
        # write your codes here
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1,320)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.fc2(x)
        
        return x

     
class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):

        # write your codes here
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(784, 30) # 28*28를 입력으로 받아 (30,1)로 출력
        self.fc2 = nn.Linear(30,20) # (30,1) 입력으로 받아 (20,1)로 출력
        self.fc3 = nn.Linear(20,10) # (20,1) 입력으로 받아 class 수대로 출력
                
        
    def forward(self, x):

        # write your codes here
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
              
        return x

#%%