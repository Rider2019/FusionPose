import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction
from torch.autograd import Variable

class Pnet2_model(nn.Module):
    def __init__(self,k=36,channel=3):
        super(Pnet2_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=16, in_channel=channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=128 + channel, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + channel, mlp=[256, 512, 1024], group_all=True)
        self.gru = nn.GRU(1024,512,bidirectional=True,batch_first=True)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
    def encode(self,xyz):
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B,1024)
        return x
    def forward(self,xyz,device):
        #print(xyz.shape)
        xyz = xyz.permute(0,1,3,2)
        B,T,_,_ = xyz.shape
        x = torch.zeros((B,T,1024)).cuda()
        for i in range(T):
            x[:,i,:] = self.encode(xyz[:,i,:,:])
        #print(x.shape)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(2, x.size(0), 512)).to(device)
        out,hid = self.gru(x,h0)
        #print(out.shape)
        x = self.dropout1(F.relu(self.fc1(out)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        #print(out.shape)
        return x.view(B,T,-1)

		
if __name__ == '__main__':
    model = Pnet2_model().cuda()
    a = torch.randn((16,30,3,256)).cuda() #B x T x C x N
    out = model(a)
    print(out.shape)
