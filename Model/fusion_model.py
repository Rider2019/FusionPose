from multiprocessing.pool import RUN
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .hrnet_32 import HigherResolutionNet
from .second_pointnet_cls import Second_Pnet_model
# from .first_pointnet_utils import PointNetEncoder
from .first_pointnet_cls import First_Pnet_model

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = 'module.', drop_prefix='', fix_loaded=True):
    success_layers, failed_layers = [], []
    def _get_params(key):
        key = key.replace(drop_prefix,'')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix!='':
                k=k.split(prefix)[1]
            success_layers.append(k)
        except:
            print('HRnet: copy param {} failed, mismatched'.format(k))
            continue
    print('missing parameters of layers:{}'.format(failed_layers))

    if fix_loaded and len(failed_layers)>0:
        print('HRnet: fixing the layers that were loaded successfully, while train the layers that failed,')
        for k in cur_state_dict.keys():
            try:
                if k in success_layers:
                    cur_state_dict[k].requires_grad=False
            except:
                print('HRnet: fixing the layer {} failed'.format(k))
    if fix_loaded:
        print('HRnet: Successfully loaded the pre trained model. The parameters of HRnet is fixed')
    else:
        print('HRnet: Successfully loaded the pre trained model. The parameters of HRnet is training')
    return success_layers

class Fusion_Model(nn.Module):
    def __init__(self,k=21*3,temporal=False,motion=False):
        super(Fusion_Model, self).__init__()
        self.pnet_encoder = First_Pnet_model().cuda()
        self.pnet_decoder = Second_Pnet_model(k,motion = motion,temporal = temporal).cuda()
        pretrained_pc = torch.load('./pretrained_model/best_model.t7')
        copy_state_dict(self.pnet_encoder.state_dict(), pretrained_pc, prefix = '', fix_loaded=False)
        copy_state_dict(self.pnet_decoder.state_dict(), pretrained_pc, prefix = '', fix_loaded=False)
        self.image_encode = HigherResolutionNet().cuda()
        pretrained_img = './pretrained_model/ROMP_HRNet32_V1.pkl'
        copy_state_dict(self.image_encode.state_dict(), torch.load(pretrained_img)['model_state_dict'], prefix = 'module.backbone.', fix_loaded=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc1 = nn.Linear(128, 128)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.conv1 = nn.Conv1d(in_channels = 8192 , out_channels = 1024 , kernel_size=1,stride=1)
        self.relu = nn.ReLU()
    def forward(self,img,pc,device):
        # raise RuntimeError(img.shape)
        B,T,W,H,C = img.shape
        B,T,num_points,_ = pc.shape
        img = img.view(B*T,W,H,C).contiguous()
        pc = pc.view(B*T,num_points,3).contiguous()
        pc_feature = self.pnet_encoder(pc,device).permute(0,2,1).contiguous()
        # raise RuntimeError(self.fc1(self.image_encode(img)).shape)
        # raise RuntimeError(self.fc1(self.image_encode(img).view(B*T,256,-1).contiguous()).shape)
        img_feature = self.relu(self.fc1(self.image_encode(img).view(B*T,256,-1).contiguous()))
        img_feature = self.transformer_encoder(img_feature)
        attention_feature = self.transformer_decoder(tgt = pc_feature,memory = img_feature)
        kp3d = self.pnet_decoder(B,T,attention_feature,device=device)
        return kp3d