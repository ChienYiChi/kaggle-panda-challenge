import torch.nn as nn
import torch


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        avg_x = self.avg(x)
        max_x = self.max(x)
        return torch.cat([avg_x, max_x], dim=1).squeeze(2).squeeze(2)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class EfficientModel(nn.Module):

    def __init__(self, c_out=5, n_tiles=36, tile_size=224, name='efficientnet-b0', strategy='stitched', head='basic'):
        super().__init__()

        from efficientnet_pytorch import EfficientNet
        m = EfficientNet.from_pretrained(name, advprop=True, num_classes=c_out, in_channels=3)
        c_feature = m._fc.in_features
        m._fc = nn.Identity()
        self.feature_extractor = m
        self.n_tiles = n_tiles
        self.tile_size = tile_size

        if strategy == 'stitched':
            if head == 'basic':
                self.head = nn.Linear(c_feature, c_out)
            elif head == 'concat':
                m._avg_pooling = AdaptiveConcatPool2d()
                self.head = nn.Linear(c_feature * 2, c_out)
            elif head == 'gem':
                m._avg_pooling = GeM()
                self.head = nn.Linear(c_feature, c_out)
        elif strategy == 'bag':
            if head == 'basic':
                self.head = BasicHead(c_feature, c_out, n_tiles)
            elif head == 'attention':
                self.head = AttentionHead(c_feature, c_out, n_tiles)

        self.strategy = strategy

    def forward(self, x):
        if self.strategy == 'bag':
            x = x.view(-1, 3, self.tile_size, self.tile_size)
        h = self.feature_extractor(x)
        h = self.head(h)
        return h

class ResnetModel(nn.Module):

    def __init__(self, c_out=5, n_tiles=36, tile_size=224, pretrained=True, strategy='stitched', head='basic'):
        super().__init__()

        from torchvision.models import resnet34
        m = resnet34(pretrained=pretrained)
        c_feature = m.fc.in_features
        m.fc = nn.Identity()
        self.feature_extractor = m
        self.n_tiles = n_tiles
        self.tile_size = tile_size

        if strategy == 'stitched':
            if head == 'basic':
                self.head = nn.Linear(c_feature, c_out)
            elif head == 'concat':
                m._avg_pooling = AdaptiveConcatPool2d()
                self.head = nn.Linear(c_feature * 2, c_out)
            elif head == 'gem':
                m._avg_pooling = GeM()
                self.head = nn.Linear(c_feature, c_out)
        elif strategy == 'bag':
            if head == 'basic':
                self.head = BasicHead(c_feature, c_out, n_tiles)
            elif head == 'attention':
                self.head = AttentionHead(c_feature, c_out, n_tiles)

        self.strategy = strategy

    def forward(self, x):
        if self.strategy == 'bag':
            x = x.view(-1, 3, self.tile_size, self.tile_size)
        h = self.feature_extractor(x)
        h = self.head(h)
        return h


class BasicHead(nn.Module):

    def __init__(self, c_in, c_out, n_tiles):
        self.n_tiles = n_tiles
        super().__init__()
        self.fc = nn.Sequential(AdaptiveConcatPool2d(),
                                nn.Linear(c_in * 2, c_out))

    def forward(self, x):

        bn, c = x.shape
        h = x.view(-1, self.n_tiles, c, 1, 1).permute(0, 2, 1, 3, 4) \
            .contiguous().view(-1, c, 1 * self.n_tiles, 1)
        h = self.fc(h)
        return h


class AttentionHead(nn.Module):

    def __init__(self, c_in, c_out, n_tiles):
        self.n_tiles = n_tiles
        super().__init__()
        self.attention_pool = AttentionPool(c_in, c_in//2)
        self.fc = nn.Linear(c_in, c_out)

    def forward(self, x):

        bn, c = x.shape
        h = x.view(-1, self.n_tiles, c)
        h = self.attention_pool(h)
        h = self.fc(h)
        return h


class AttentionPool(nn.Module):

    def __init__(self, c_in, d):
        super().__init__()
        self.lin_V = nn.Linear(c_in, d)
        self.lin_w = nn.Linear(d, 1)

    def compute_weights(self, x):
        key = self.lin_V(x)  # b, n, d
        weights = self.lin_w(torch.tanh(key))  # b, n, 1
        weights = torch.softmax(weights, dim=1)
        return weights

    def forward(self, x):
        weights = self.compute_weights(x)
        pooled = torch.matmul(x.transpose(1, 2), weights).squeeze(2)   # b, c, n x b, n, 1 => b, c, 1
        return pooled


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.parameter.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + \
               'eps=' + str(self.eps) + ')'


class QWKLoss(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        grid = torch.repeat_interleave(torch.arange(0, n_class).reshape(n_class, 1), repeats=n_class, dim=1)
        self.weights = ((grid - grid.T) ** 2) / float((n_class - 1) ** 2)

    def forward(self, logits, y_true):
        y_pred = logits.softmax(1)

        weights = self.weights.to(logits.device)

        nom = torch.matmul(y_true, weights)  # N, C * C, C = N, C
        nom = nom * y_pred  # N, C * C, N = N, N
        nom = nom.sum()

        denom = y_pred.sum(0, keepdims=True)
        n_hat = y_true.sum(0, keepdims=True) / y_true.shape[0]
        denom = torch.matmul(n_hat.T, denom) * weights
        denom = denom.sum()

        # gradient descent minimizes therefore return 1 - kappa instead of kappa = 1 - nom/denom
        return nom / denom
