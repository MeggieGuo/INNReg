import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        self.groups = groups 

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        # weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        weight = self.conv2(self.relu(self.conv1(self.pool(x))))
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x

class DWBlock(nn.Module):

    def __init__(self, dim, window_size, dynamic=False, inhomogeneous=False, heads=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.dynamic = dynamic 
        self.inhomogeneous = inhomogeneous
        self.heads = heads
        
        # pw-linear
        self.conv0 = nn.Conv2d(1, dim, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(dim)

        if dynamic and not inhomogeneous:
            self.conv = DynamicDWConv(dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)
        elif dynamic and inhomogeneous:
            print(window_size, heads)
            # self.conv = IDynamicDWConv(dim, window_size, heads)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)
       
        
        self.bn = nn.BatchNorm2d(dim)
        self.relu=nn.ReLU(inplace=True)
                
        # pw-linear
        self.conv2=nn.Conv2d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

        self.conv3=nn.Conv2d(dim, 1, 1, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        
        x = self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        
        x = self.conv2(x)
        x=self.bn2(x)

        x = self.conv3(x)
        
        # x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # x = self.conv0(x)
        flops += N * self.dim * self.dim
        # x = self.conv(x)
        if self.dynamic and not self.inhomogeneous:
            flops += (N * self.dim + self.dim * self.dim / 4 + self.dim / 4 * self.dim * self.window_size * self.window_size)
        elif self.dynamic and self.inhomogeneous:
            flops += (N * self.dim * self.dim / 4 + N * self.dim / 4 * self.dim / self.heads * self.window_size * self.window_size)
        flops +=  N * self.dim * self.window_size * self.window_size
        #  x = self.conv2(x)
        flops += N * self.dim * self.dim
        #  batchnorm + relu
        flops += 8 * self.dim * N
        return flops

#### DWDB
class DWDBNet(nn.Module):
    def __init__(self, channel_in, channel_out, dim, window_size, init='xavier', gc=32, bias=True):
        super(DWDBNet, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc-1, 3, 1, 1, bias=bias)

        self.dwconv = DynamicDWConv(128, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)
        self.conv5 = nn.Conv2d(128, channel_out, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # print(torch.cat((x, x1, x2, x3, x4), 1).shape)
        x45 = self.lrelu(self.dwconv(torch.cat((x, x1, x2, x3, x4), 1)))
        x5 = self.conv5(x45)

        return x5

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

