import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .second_pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from torch.autograd import Variable

class Second_Pnet_model(nn.Module):
    def __init__(self,k=36, channel=3,temporal = False,motion = False,smpl = False):
        super(Second_Pnet_model, self).__init__()
        kp_num = int(k/3)
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 63)
        self.gru = nn.GRU(1024,512,bidirectional=True,batch_first=True)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.temporal = temporal
        self.motion = motion
        self.smpl = smpl
        
        self.motion_fc = nn.Linear(256, k)
        self.theta_fc = nn.Sequential(
            nn.Linear(256+k, 128),
            nn.ReLU(),   
            nn.Linear(128, 72),
        )
        self.partarial = nn.Sequential(
            nn.Linear(256, kp_num*32),
            # nn.BatchNorm1d(kp_num*32),
            nn.ReLU(),   
            nn.Linear(kp_num*32, kp_num*64),
            nn.ReLU()      
        )
        self.fc_kp = nn.Linear(kp_num*67, k)

    def encode(self, x):
        x = self.feat(x)
        return x
    
    def forward(self,B,T,x,device):
        
        x = x.permute(0,2,1)
        x = self.encode(x).view(B,T,1024).contiguous()
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(2, x.size(0), 512)).to(device)
        out,hid = self.gru(x,h0)
        x = F.relu((self.fc1(out)))
        x = F.relu((self.dropout(self.fc2(x))))

        if self.temporal:
            x_kp = self.fc3(x)

            x_part = self.partarial(x)
            # print(x_part.shape,x_kp.shape)
            x_kp2 = self.fc_kp(torch.cat((x_part,x_kp),2))

            return x_kp.view(B,T,-1),x_part.view(B,T,-1),x_kp2.view(B,T,-1)
        elif self.motion:
            x1 = self.fc3(x)
            x2 = self.motion_fc(x)
            return x1.view(B,T,-1),x2.view(B,T,-1)
        elif self.smpl:
            x1 = self.fc3(x)
            x2 = self.theta_fc(torch.cat((x,x1),2))
            return x1.view(B,T,-1),x2.view(B,T,-1)
        else:
            x = self.fc3(x)

        # print(x.shape)
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
    model = Second_Pnet_model().cuda()
    a = torch.randn((16,30,128,3)).cuda() #B x T x C x N
    out = model(a,'cuda')
    print(out.shape)
