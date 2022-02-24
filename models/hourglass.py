import torch.nn as nn
import torch.nn.functional as F

# from .preresnet import BasicBlock, Bottleneck


__all__ = ['HourglassNet']

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, mobile=False):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        if mobile:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=True, groups=planes)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, nModules, planes, depth, mobile):
        super(Hourglass, self).__init__()
        self.mobile = mobile
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, nModules, planes, depth)

    def _make_residual(self, block, nModules, planes):
        layers = []
        for i in range(0, nModules):
            layers.append(block(planes*block.expansion, planes, mobile=self.mobile))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, nModules, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, nModules, planes))
            if i == 0:
                res.append(self._make_residual(block, nModules, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block=Bottleneck, nStack=4, nModules=4, nHgDepth=4, nFeats=128, nClasses=16, mobile=False):
        super(HourglassNet, self).__init__()

        self.mobile = mobile
        self.inplanes = 64
        self.num_feats = nFeats
        self.num_stacks = nStack
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(self.num_stacks):
            hg.append(Hourglass(block, nModules, self.num_feats, 4, self.mobile))
            res.append(self._make_residual(block, self.num_feats, nModules))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, nClasses, kernel_size=1, bias=True))
            if i < self.num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(nClasses, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)


    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.mobile))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, mobile=self.mobile))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)  

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        return out


if __name__=="__main__":
    model = HourglassNet(Bottleneck, nClasses=68)
    from torchsummary import summary
    import torch
    torch.save(model, "./hourglass.pth")
    inp = torch.rand(1,3,256,256)
    # out = model(inp)
    summary(model, inp)
    import ipdb; ipdb.set_trace()