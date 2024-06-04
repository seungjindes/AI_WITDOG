import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F

import warnings


warnings.filterwarnings("ignore")

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class SER_AlexNet(nn.Module):

    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(SER_AlexNet, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool
        self.classifier = model.classifier
        

        # if in_ch != 3:
        #     self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        #     init_layer(self.features[0])

        self.classifier[6] = nn.Linear(4096, num_classes)

        self._init_weights(pretrained=pretrained)
        
        #print('\n<< SER AlexNet Finetuning model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        # x = self.avgpool(x)
        x = F.avg_pool2d(x, kernel_size=(2,3) , padding = (0,1) , stride = (1,3))
        x_ = torch.flatten(x, 1)
        out = self.classifier(x_)

        return x, out

    def _init_weights(self, pretrained=True):

        init_layer(self.classifier[6])

        if pretrained == False:
            init_layer(self.features[0])
            init_layer(self.features[3])
            init_layer(self.features[6])
            init_layer(self.features[8])
            init_layer(self.features[10])
            init_layer(self.classifier[1])
            init_layer(self.classifier[4])



class SER_ResNet(nn.Module):

    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(SER_ResNet, self).__init__()


        self.model = torchvision.models.resnet101(pretrained = pretrained)
        self.out_cls = torch.nn.Linear(1000 , num_classes , bias = True)
        
    
        # if in_ch != 3:
        #     self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        #     init_layer(self.features[0])
        init_layer(self.out_cls)
        


    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x_ = torch.flatten(x, 1)
        x_ = self.model.fc(x_)
        out = self.out_cls(x_)
        return x , out
    
    



class SER_EfficientNet(nn.Module):

    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(SER_EfficientNet, self).__init__()


        self.model = torchvision.models.efficientnet_v2_s(pretrained = pretrained)

    
        self.out_cls = torch.nn.Linear(1000 , num_classes , bias = True)
        
        # if in_ch != 3:
        #     self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        #     init_layer(self.features[0])
        init_layer(self.out_cls)
        


    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x_ = torch.flatten(x, 1)

        x_ = self.model.classifier(x_)
        out = self.out_cls(x_)

        return x , out
    

class SER_VIT(nn.Module):

    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(SER_VIT, self).__init__()


        self.model = torchvision.models.vit_b_16(pretrained = pretrained)

        self.input_gate = torch.nn.Conv2d(3 , 64 , kernel_size = (12,24) , stride = (1,2))

    
        self.out_cls = torch.nn.Linear(1000 , num_classes , bias = True)
        
        # if in_ch != 3:
        #     self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        #     init_layer(self.features[0])
        init_layer(self.out_cls)
        
    # vit의 경우 input이 224 224로 들어와야한다.
        
    def forward(self, x):
 
        print(x.shape)
        x = self.input_gate(x)
        print(x.shape)
        raise Exception       
        x = self.model(x)
        
        # x = self.model.conv_proj(x)

        # x - self.model.encoder.layers.encoder_layer_0.ln_1(x)
        
        raise Exception

        x = self.model.encoder(x)
        x_ = torch.flatten(x, 1)
        x_ = self.model.heads(x_)
        out = self.out_cls(x_)

        return x , out
    



# class Alex_coc(nn.Module):
#     def __int__(self , pretrained = Ture):
#         super(Ael)

        
        