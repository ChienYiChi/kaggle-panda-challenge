import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet


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


if __name__=='__main__':
    pass
