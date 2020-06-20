import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models
import torch.nn.init as init
import math
from torch.autograd import Variable
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


class Resnext50Tiles(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', num_classes=6):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(2*nc,512),
                            Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,num_classes))
        
    def forward(self, x):
        """
        Args:
            x (batch,N,3,h,w):
        """
        shape = x[0].shape
        n = shape[0]
        x = x.view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        x = self.head(x)
        return x

class EfficientnetTiles(nn.Module):
    def __init__(self, arch='efficientnet-b0', num_classes=6):
        super().__init__()
        self.base = EfficientNet.from_pretrained(arch)
        self.in_features = self.base._fc.in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(2*self.in_features,512),
                            Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,num_classes))
        
    def forward(self, x):
        """
        Args:
            x (batch,N,3,h,w):
        """
        shape = x[0].shape
        n = shape[0]
        x = x.view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.base.extract_features(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        x = self.head(x)
        return x


class EnetV1(nn.Module):
    def __init__(self, backbone='efficientnet-b0', num_classes=6):
        super(EnetV1, self).__init__()
        self.enet = EfficientNet.from_pretrained(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, num_classes)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


class EnetV2(nn.Module):
    def __init__(self, backbone='efficientnet-b0', num_classes=6):
        super(EnetV2, self).__init__()
        self.enet = EfficientNet.from_pretrained(backbone)
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                nn.Flatten(),
                                nn.Linear(2*self.enet._fc.in_features,num_classes))
    def forward(self, x):
        x = self.enet.extract_features(x)
        x = self.head(x)
        return x


class Resnext50(nn.Module):
    def __init__(self,arch='resnext50_32x4d_ssl',num_classes=6):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.model.fc = nn.Linear(self.model.fc.in_features,num_classes)
        
    def forward(self,x):
        x = self.model(x)
        return x 

    
class SEResNeXt(nn.Module):
    def __init__(self,arch='se_resnext50_32x4d',num_classes=1,pretrained='imagenet'):
        super().__init__()
        self.model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=pretrained)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features,num_classes)
    
    def forward(self,x):
        x = self.model(x)
        return x 


class NetVLAD(nn.Module):
    def __init__(self, feature_size, max_frames,cluster_size, add_bn=False, truncate=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size / 2 if truncate else feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.batch_norm = nn.BatchNorm1d(cluster_size, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(self.feature_size, self.cluster_size)
        self.softmax = nn.Softmax(dim=1)
        self.cluster_weights2 = nn.Parameter(torch.FloatTensor(1, self.feature_size,
                                                               self.cluster_size))
        self.add_bn = add_bn
        self.truncate = truncate
        self.first = True
        self.init_parameters()

    def init_parameters(self):
        init.normal_(self.cluster_weights2, std=1 / math.sqrt(self.feature_size))

    def forward(self, reshaped_input):
        random_idx = torch.bernoulli(torch.Tensor([0.5]))
        if self.truncate:
            if self.training == True:
                reshaped_input = reshaped_input[:, :self.feature_size].contiguous() if random_idx[0]==0 else reshaped_input[:, self.feature_size:].contiguous()
            else:
                if self.first == True:
                    reshaped_input = reshaped_input[:, :self.feature_size].contiguous()
                else:
                    reshaped_input = reshaped_input[:, self.feature_size:].contiguous()
        activation = self.linear(reshaped_input)
        if self.add_bn:
            activation = self.batch_norm(activation)
        activation = self.softmax(activation).view([-1, self.max_frames, self.cluster_size])
        a_sum = activation.sum(-2).unsqueeze(1)
        a = torch.mul(a_sum, self.cluster_weights2)
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = reshaped_input.view([-1, self.max_frames, self.feature_size])
        vlad = torch.matmul(activation, reshaped_input).permute(0, 2, 1).contiguous()
        vlad = vlad.sub(a).view([-1, self.cluster_size * self.feature_size])
        if self.training == False:
            self.first = 1 - self.first
        return vlad


class Resnext50wNetVLAD(nn.Module):
    def __init__(self, num_clusters,num_tiles,num_classes,arch='resnext50_32x4d_ssl'):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-1])       
        self.nc = list(m.children())[-1].in_features
        self.netvlad = NetVLAD(cluster_size=num_clusters,max_frames=num_tiles,
                    feature_size=self.nc,truncate=False)
        self.fc = nn.Linear(num_clusters*self.nc,num_classes)
        
    def forward(self, x):
        """
        Args:
            x (batch,N,3,h,w):
        """
        batch = x.shape[0]
        shape = x[0].shape
        n = shape[0]
        x = x.view(-1,shape[1],shape[2],shape[3]) #x: bs*num_tiles x 3 x H x W
        x = self.enc(x) #x: bs*num_tiles x nc
        x = x.view(batch,n,self.nc)
        x = self.netvlad(x)
        x = self.fc(x)
        return x


class ResnetwNetVLAD(nn.Module):
    def __init__(self, num_clusters,num_tiles,num_classes,pretrained=True):
        super().__init__()
        m = models.resnet34(pretrained=pretrained)
        self.enc = nn.Sequential(*list(m.children())[:-1])       
        self.nc = list(m.children())[-1].in_features
        self.netvlad = NetVLAD(cluster_size=num_clusters,max_frames=num_tiles,
                    feature_size=self.nc,truncate=False)
        self.fc = nn.Linear(num_clusters*self.nc,num_classes)
        
    def forward(self, x):
        """
        Args:
            x (batch,N,3,h,w):
        """
        batch = x.shape[0]
        shape = x[0].shape
        n = shape[0]
        x = x.view(-1,shape[1],shape[2],shape[3]) #x: bs*num_tiles x 3 x H x W
        x = self.enc(x) #x: bs*num_tiles x nc
        x = x.view(batch,n,self.nc)
        x = self.netvlad(x)
        x = self.fc(x)
        return x


class EnetNetVLAD(nn.Module):
    def __init__(self, num_clusters,num_tiles,num_classes,arch='efficientnet-b0'):
        super().__init__()
        self.base = EfficientNet.from_pretrained(arch)
        self.nc = self.base._fc.in_features
        self.tile = nn.Sequential(
            nn.BatchNorm2d(self.nc,eps=0.001,momentum=0.01),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
        )    
        self.netvlad = NetVLAD(cluster_size=num_clusters,max_frames=num_tiles,
                    feature_size=self.nc,truncate=False)
        self.fc = nn.Linear(num_clusters*self.nc,num_classes)

    def forward(self, x):
        """
        Args:
            x (batch,N,3,h,w):
        """
        batch = x.shape[0]
        shape = x[0].shape
        n = shape[0]
        x = x.view(-1,shape[1],shape[2],shape[3]) #x: bs*num_tiles x 3 x H x W
        x = self.base.extract_features(x) #x: bs*num_tiles x nc
        x = self.tile(x)
        x = x.view(batch,n,self.nc)
        x = self.netvlad(x)
        x = self.fc(x)
        return x

if __name__=='__main__':
    x = torch.rand(4,36,3,128,128).cuda()
    model = EnetNetVLAD(num_clusters=6,num_tiles=36,num_classes=6).cuda()
    output = model(x)
    print(output.size())

    
