import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.stride = 1
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, stride=1)
        # self.conv21 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
        # self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.conv21(out)
        # out = self.conv22(out)
        out = self.bn2(out)
        
        if self.project:
            residual = self.project(x)
        out += residual
        out = self.relu(out)
        return out

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleAttention, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x  # channel-wise attention
        x = self.sa(x) * x  # spatial attention
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, init_type='normal'):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        if init_type == 'normal':
            nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))
        elif init_type == 'xaiver':
            nn.init.xavier_normal_(self.cheby_coeffs)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(self.cheby_coeffs)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = torch.tanh(x)
        
        if len(x.shape) == 2:
            
            cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
            if self.degree > 0:
                cheby[:, :, 1] = x
            for i in range(2, self.degree + 1):
                cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()

            y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)  # shape = (batch_size, outdim)

            return y
        
        elif len(x.shape) == 3:
            
            cheby = torch.ones(x.shape[0], x.shape[1], self.inputdim, self.degree + 1, device=x.device)
            if self.degree > 0:
                cheby[:, :, :, 1] = x
            for i in range(2, self.degree + 1):
                cheby[:, :, :, i] = 2 * x * cheby[:, :, :, i - 1].clone() - cheby[:, :, :, i - 2].clone()

            y = torch.einsum('bnid,iod->bno', cheby, self.cheby_coeffs)  # shape = (batch_size, outdim)

            return y

class ChebyKAN(nn.Module):
    def __init__(self, width=None, degree=4, init_type='normal'):
        super(ChebyKAN, self).__init__()

        self.width = width 
        self.depth = len(width) - 1

        act_fun, norm_fun = [], []
        for l in range(self.depth):
            act_fun.append(ChebyKANLayer(self.width[l], self.width[l + 1], degree, init_type=init_type)) 
            norm_fun.append(nn.LayerNorm(self.width[l + 1]))

        self.act_fun = nn.ModuleList(act_fun)

        self.degree = degree
        
    def forward(self, x):

        for l in range(self.depth):
            x = self.act_fun[l](x)  

        return x

class MineralClassifier(nn.Module):
    def __init__(self,num_classes=2,in_dim=2,degree=12,init_type='normal'):
        super(MineralClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.attention = MultiScaleAttention(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, num_classes) # Classification head
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.attention(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x) # Single-branch selection adds a classification head
        return x


class dual_tower(nn.Module):
    def __init__(self, in_dim, num_classes, pretrained=None, degree=6, init_type='normal') -> None:
        super(dual_tower, self).__init__()
        self.model_low = MineralClassifier(num_classes=num_classes, in_dim=in_dim)
        self.model_high = MineralClassifier(num_classes=num_classes, in_dim=in_dim)

        # Pre-training parameter Settings
        if pretrained is not None:
            assert os.path.exists(pretrained), "weights file: '{}' not exist.".format(pretrained)
            weights_dict = torch.load(pretrained, map_location='cpu')
            
            self.model_low.load_state_dict(weights_dict,strict=False)
            self.model_high.load_state_dict(weights_dict,strict=False)

        self.classifier = ChebyKAN(width=[128,64,32,16,8,4,2], degree=degree, init_type=init_type)

    def forward(self, input_low, input_high):
        low = input_low + input_high
        high = input_low - input_high

        x_low = self.model_low(low)
        x_high = self.model_high(high)

        x = x_low + x_high

        x_fused_cls = self.classifier(x)

        return x_fused_cls
    

#这边又分开搞了两个类，代码和之前相同
class dual_tower_low(nn.Module):
    def __init__(self, in_dim, num_classes,pretrained=None) -> None:
        super(dual_tower_low, self).__init__()
        self.model_low = MineralClassifier(num_classes=num_classes, in_dim=in_dim)
        if pretrained is not None:
            params = torch.load(pretrained, map_location='cpu')['state_dict']
            self.model_low.load_state_dict(params,strict=False)
    
    def forward(self, input_low, input_high):
        x_low = self.model_low(input_low)
        return x_low
    
class dual_tower_high(nn.Module):
    def __init__(self, in_dim, num_classes,pretrained=None) -> None:
        super(dual_tower_high, self).__init__()
        self.model_high = MineralClassifier(num_classes=num_classes, in_dim=in_dim)
        if pretrained is not None:
            params = torch.load(pretrained, map_location='cpu')['state_dict']
            self.model_high.load_state_dict(params,strict=False)
    
    def forward(self, input_low, input_high):
        x_high = self.model_high(input_high)
        return x_high
    
class dual_tower_vis(nn.Module):
    def __init__(self, in_dim, num_classes, pretrained=None, degree=6, init_type='normal') -> None:
        super(dual_tower_vis, self).__init__()
        self.model_low = MineralClassifier(num_classes=num_classes, in_dim=in_dim)
        self.model_high = MineralClassifier(num_classes=num_classes, in_dim=in_dim)
        if pretrained is not None:
            params = torch.load(pretrained, map_location='cpu')['state_dict']
            self.model_low.load_state_dict(params,strict=False)
            self.model_high.load_state_dict(params,strict=False)
        self.classifier = ChebyKAN(width=[128,64,32,16,8,4,2], degree=degree, init_type=init_type)
    
    def forward(self, inputs):
        input_low, input_high = inputs[:,0,:,:], inputs[:,1,:,:]
        input_low = input_low.unsqueeze(1)
        input_high = input_high.unsqueeze(1)
        low = input_low + input_high
        high = input_low - input_high
        x_low = self.model_low(low)
        x_high = self.model_high(high)
        x = x_low + x_high
        x_fused_cls = self.classifier(x)
        return x_fused_cls

def create_model(args):
    assert args.modelName in ['dual_tower', 'dual_tower_high', 'dual_tower_low', 'dual_tower_vis']
    if args.modelName == 'dual_tower': 
        return dual_tower(args.in_dim, args.num_classes, args.pretrained)
    elif args.modelName == 'dual_tower_high':
        return dual_tower_high(args.in_dim, args.num_classes, args.pretrained)
    elif args.modelName == 'dual_tower_low':
        return dual_tower_low(args.in_dim, args.num_classes, args.pretrained)
    
    elif args.modelName == 'dual_tower_vis':
        return dual_tower_vis(args.in_dim, args.num_classes, args.pretrained)
    else:
        raise NotImplementedError

