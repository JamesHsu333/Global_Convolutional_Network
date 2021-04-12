import model.net as net

from torchsummary import summary

model = net.FCN_GCN(20+1).cuda()
summary(model, (3, 224, 224))