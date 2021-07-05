import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Net,self).__init__()
        self.conv1 = nn.Conv1d(in_channels,16,kernel_size=1)#512
        self.conv3 = nn.Conv1d(16, 32, kernel_size=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, out_channels,kernel_size=1)#512
        self.bn1 = nn.BatchNorm1d(16)#512
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bnm1 = nn.BatchNorm1d(256)
        self.bnm2 = nn.BatchNorm1d(128)
        self.bnm3 = nn.BatchNorm1d(512)
        #self.fc1 = nn.Linear(2000* out_channels, 512)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(2000*out_channels, 512)
        self.fc4 = nn .Linear(512,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        #out = torch.max(out, 2, keepdim=True)[0]
        #out = out.reshape(-1, 512)
        out = out.reshape(out.size(0), -1)
        #out =self.fc1(out)

        out = F.relu(self.bnm3(self.dropout(self.fc1(out))))
        out = F.relu(self.bnm1(self.dropout(self.fc4(out))))
        out = F.relu(self.bnm2(self.dropout(self.fc2(out))))
        out = self.fc3(out)
        return out
'''
net = torch.nn.Sequential(
    nn.Linear(8000,20),
    torch.nn.ReLU(),
    nn.Linear(20,20),
    torch.nn.ReLU(),
    nn.Linear(20,6)
)
'''
'''# net = Net(8000,20,6)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.05)
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(2000):
    prediction = net(x)
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.05)

torch.save(net,'net.pkl')
torch.save(net.state_dict(),'net_parameter.pkl')
'''
'''
net1 = torch.load('net.pkl')
net2 = torch.nn.Sequential(
    nn.Linear(1,20),
    torch.nn.ReLU(),
    nn.Linear(20,20),
    torch.nn.ReLU(),
    nn.Linear(20,1)
)
net2.load_state_dict(torch.load('net_parameter.pkl'))

prediction1 = net1(x)
prediction2 = net2(x)

plt.figure(1,figsize=(10,3))
plt.subplot(121)
plt.title('net1')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction1.data.numpy(), 'r-', lw=5)
# plt.show()

plt.figure(1,figsize=(10,3))
plt.subplot(122)
plt.title('net2')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction2.data.numpy(), 'r-', lw=5)
plt.show()
'''