import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models
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


# class NetVLAD(nn.Module):
#     """NetVLAD layer implementation"""

#     def __init__(self, num_clusters, num_tiles,feature_size, alpha=100.0,
#                 normalize_input=True):
#         """
#         Args:
#             num_clusters : int
#                 The number of clusters
#             num_tiles : int
#                 Dimension of descriptors
#             alpha : float
#                 Parameter of initialization. Larger value is harder assignment.
#             normalize_input : bool
#                 If true, descriptor-wise L2 normalization is applied to input.
#         """
#         super(NetVLAD, self).__init__()
#         self.num_clusters = num_clusters
#         self.num_tiles = num_tiles
#         self.alpha = alpha
#         self.normalize_input = normalize_input
#         self.fc = nn.Linear(feature_size,num_clusters)
#         self.centroids = nn.Parameter(torch.rand(num_clusters,num_tiles))
        
#     def forward(self, x):
#         """
#         Args:
#             x: [batch,num_tiles,feature_size]
#         """
#         if self.normalize_input:
#             x = F.normalize(x, p=2, dim=1)  # across descriptor dim

#         # soft-assignment
#         soft_assign = self.fc(x) #[batch,num_tiles,num_clusters]
#         soft_assign = F.softmax(soft_assign, dim=-1).permute(0,2,1)
        
#         # calculate residuals to each clusters
#         residual = x.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
#             self.centroids.expand(x.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
#         residual *= soft_assign.unsqueeze(3)
#         vlad = residual.sum(dim=-1)

#         vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
#         vlad = vlad.view(x.size(0), -1)  # flatten
#         vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

#         return vlad

import torch.nn.init as init
import math
from torch.autograd import Variable


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


class selfAttn(nn.Module):
    def __init__(self, feature_size, time_step, hidden_size, num_desc):
        super(selfAttn, self).__init__()
        self.linear_1 = nn.Linear(feature_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_desc)
        self.num_desc = num_desc
        #self.init_weights()
    
    def init_weights(self):
        self.linear_1.weight.data.uniform_(-0.1, 0.1)
        #self.linear_2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, model_input):   # (batch_size, time_step, feature_size)
        reshaped_input = model_input  # (batch_size, feature_step, time_step)
        s1 = F.tanh(self.linear_1(reshaped_input))  # (batch_size, feature_size, hidden_size)
        A = F.sigmoid(self.linear_2(s1))
        M = torch.bmm(model_input.permute(0, 2, 1), A).permute(0, 2, 1).contiguous()  # (batch_size, time_step, num_desc)
        AAT = torch.bmm(A.permute(0, 2, 1), A)
        I = Variable(torch.eye(self.num_desc)).cuda()
        P = torch.norm(AAT - I, 2)
        penal = P * P / model_input.shape[0]
        return M


class MoeModel(nn.Module):
    def __init__(self, num_classes, feature_size, num_mixture=2):
        super(MoeModel, self).__init__()
        self.gating = nn.Linear(feature_size, num_classes * (num_mixture+1))
        self.expert = nn.Linear(feature_size, num_classes * num_mixture)
        self.num_mixture = num_mixture
        self.num_classes = num_classes
    def forward(self, model_input):
        gate_activations = self.gating(model_input)
        gate_dist = nn.Softmax(dim=1)(gate_activations.view([-1, self.num_mixture + 1]))
        expert_activations = self.expert(model_input)
        expert_dist = nn.Softmax(dim=1)(expert_activations.view([-1, self.num_mixture]))
        probabilities_by_class_and_batch = torch.sum(
            gate_dist[:, :self.num_mixture] * expert_dist, 1)
        return probabilities_by_class_and_batch.view([-1, self.num_classes])


class NetVLADModelLF(nn.Module):
    def __init__(self, cluster_size, max_frames, feature_size, hidden_size, num_classes, add_bn=False, use_moe=True, truncate=True, attention=False, use_VLAD=False):
        super(NetVLADModelLF, self).__init__()
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.video_NetVLAD = NetVLAD(self.feature_size, 100, self.cluster_size, truncate=truncate, add_bn=add_bn) if use_VLAD else None
        self.batch_norm_input = nn.BatchNorm1d(feature_size, eps=1e-3, momentum=0.01)
        self.batch_norm_activ = nn.BatchNorm1d(hidden_size, eps=1e-3, momentum=0.01)
        if use_VLAD:
            self.linear_1 = nn.Linear(cluster_size * self.feature_size / 2, hidden_size) if truncate else nn.Linear(cluster_size * self.feature_size, hidden_size)
        else:
            self.linear_1 = nn.Linear(self.feature_size, hidden_size)
        self.relu = nn.ReLU6()
        self.linear_2 = nn.Linear(hidden_size, num_classes)
        self.s = nn.Sigmoid()
        self.moe = MoeModel(num_classes, hidden_size) if use_moe else None
        self.Attn = selfAttn(feature_size, max_frames, 4096, 100) if attention else None
        self.add_bn = add_bn
        self.truncate = truncate
        self.use_moe = use_moe
        self.attention = attention
        self.use_VLAD = use_VLAD

    def forward(self, model_input):
        #import pdb;pdb.set_trace()
        reshaped_input = model_input.view([-1, self.feature_size])
        if self.add_bn:
            reshaped_input = self.batch_norm_input(reshaped_input)
        if self.attention:
            model_input = self.Attn(reshaped_input.view([-1, self.max_frames, self.feature_size]))
        if self.use_VLAD:
            output = self.video_NetVLAD(model_input.view([-1, self.feature_size]))
        else:
            output, _ = torch.max(model_input.view([model_input.shape[0], -1, self.feature_size]), dim=1)
        if self.add_bn:
            activation = self.batch_norm_activ(self.linear_1(output))
            #activation = self.linear_1(output)
        activation = self.relu(activation)
        if self.use_moe:
            logits = self.moe(activation)
        else:
            logits = self.s(self.linear_2(activation))
        return logits


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
    x = torch.rand(4,12,3,128,128)
    model = EnetNetVLAD(num_clusters=6,num_tiles=12,num_classes=6)
    output = model(x)

    