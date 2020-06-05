import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from efficientnet_pytorch import EfficientNet


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)

# Reference: https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


class Resnext50Tiles(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(2*nc,512),
                            Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        # self.head = nn.Sequential(GeM(),nn.Flatten(),nn.Linear(nc,512),
        #                     Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))                    
        
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
    def __init__(self, arch='efficientnet-b0', n=6):
        super().__init__()
        self.base = EfficientNet.from_pretrained(arch)
        self.in_features = self.base._fc.in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(2*self.in_features,512),
                            Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        
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


class Resnext50TilesGRU(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-1])       
        self.nc = list(m.children())[-1].in_features
        self.gru = torch.nn.GRU(self.nc,self.nc,batch_first=True)
        self.fc = nn.Linear(self.nc,6)
    def forward(self, x):
        """
        Args:
            x (batch,N,3,h,w):
        """
        batch = x.shape[0]
        shape = x[0].shape
        n = shape[0]
        x = x.view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        #x: bs*N x nc
        x = x.view(batch,n,self.nc)
        _,x = self.gru(x)#[1,batch,nc]
        x = self.fc(x.squeeze(0))
        return x