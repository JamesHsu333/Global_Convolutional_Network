import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

from model.GCN import GCN
from model.BR import BR

resnet = torchvision.models.resnet50(pretrained=True)

class FCN_GCN(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(FCN_GCN, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048
        
        self.gcn1 = GCN(256,self.num_classes) #gcn_i after layer-1
        self.gcn2 = GCN(512,self.num_classes)
        self.gcn3 = GCN(1024,self.num_classes)
        self.gcn4 = GCN(2048,self.num_classes)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)
        self.br8 = BR(num_classes)
        self.br9 = BR(num_classes)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c,in_c,3,padding=1,bias=False),
            nn.BatchNorm2d(in_c/2),
            nn.ReLU(inplace=True),
            #nn.Dropout(.1),
            nn.Conv2d(in_c/2, self.num_classes, 1),
            )

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm1 = self.br1(self.gcn1(fm1))
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.interpolate(self.br5(gc_fm3 + gc_fm4), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = F.interpolate(self.br6(gc_fm2 + gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm1 = F.interpolate(self.br7(gc_fm1 + gc_fm2), pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.interpolate(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        out = F.interpolate(self.br9(gc_fm1), input.size()[2:], mode='bilinear', align_corners=True)

        return out

def loss_fn(outputs, labels):
    return nn.BCEWithLogitsLoss()(outputs, labels)

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    def forward(self, logits, targets):
        smooth = 1.
        logits = torch.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class SoftInvDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftInvDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = torch.sigmoid(logits)
        iflat = 1 - logits.view(-1)
        tflat = 1 - targets.view(-1)
        intersection = (iflat * tflat).sum()
    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

loss_fns = {
    'BinaryCrossEntropy': loss_fn,
    'SoftDiceLoss': SoftDiceLoss(),
    'SoftInvDiceLoss': SoftInvDiceLoss()
}

class ConfusionMatrix:
    def __init__(self, outputs, labels, class_nums):
        self.outputs = outputs
        self.labels = labels
        self.class_nums = class_nums
    def construct(self):
        self.outputs = self.outputs.flatten()
        self.outputs_count = np.bincount(self.outputs, minlength=self.class_nums)
        self.labels = self.labels.flatten()
        self.labels_count = np.bincount(self.labels, minlength=self.class_nums)

        tmp = self.labels * self.class_nums + self.outputs

        self.cm = np.bincount(tmp, minlength=self.class_nums*self.class_nums)
        self.cm = self.cm.reshape((self.class_nums, self.class_nums))

        self.Nr = np.diag(self.cm)
        self.Dr = self.outputs_count + self.labels_count - self.Nr
    def mIOU(self):
        iou = self.Nr / self.Dr
        miou = np.nanmean(iou)
        return miou

def mIOU(outputs, labels, class_nums):
    for _, (output, label) in enumerate(zip(outputs, labels)):
        output = output.transpose(1,2,0)
        label = label.transpose(1,2,0)
        output = np.argmax(output, axis=2)
        label = np.argmax(label, axis=2)
        cm = ConfusionMatrix(output, label, class_nums)
        cm.construct()
        return cm.mIOU()

metrics = {
    'mIOU': mIOU,
}