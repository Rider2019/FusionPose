import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .first_pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from torch.autograd import Variable

class First_Pnet_model(nn.Module):
    def __init__(self,k=36, channel=3,fusion = False):
        super(First_Pnet_model, self).__init__()

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.gru = nn.GRU(1024,512,bidirectional=True,batch_first=True)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fusion = fusion

    def encode(self, x):
        #x = x.permute(0,2,1)
        x = self.feat(x)
        #x = F.log_softmax(x, dim=1)
        return x
    
    def forward(self,x,device):
        xyz = x.permute(0,2,1)
        return self.encode(xyz)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(2, x.size(0), 512)).to(device)
        out,hid = self.gru(x,h0)
        #print(out.shape)

        x = F.relu((self.fc1(out)))
        x = F.relu((self.dropout(self.fc2(x))))
        if self.fusion:
            return x
        x = self.fc3(x)
        #print(x.shape)
        return x.view(B,T,-1)

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

if __name__ == '__main__':
    model = Pnet_model().cuda()
    a = torch.randn((16,30,128,3)).cuda() #B x T x C x N
    out = model(a,'cuda')
    print(out.shape)
