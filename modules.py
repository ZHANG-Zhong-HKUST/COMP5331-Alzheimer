import torch
import torch.nn as nn

class SENet(nn.Module):
    'Squeeze-and-Excitation networks'

    def __init__(self, c_in, dim_red_factor=16) -> None:
        super().__init__()
        c_hidden = int(c_in/dim_red_factor)
        self.global_avg_pooling = lambda x:torch.mean(x, (1, 2, 3), keepdim=True)
        self.linear_1 = torch.nn.Linear(c_in, c_hidden)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(c_hidden, c_in)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, inputs: torch.Tensor):
        inputs = inputs.permute((0, 2, 3, 4, 1))
        x = self.global_avg_pooling(inputs)
        x = self.linear_2(self.relu(self.linear_1(x)))
        attns = self.sigmoid(x)
        ret = torch.mul(inputs, attns).permute((0, 4, 1, 2, 3))
        
        return ret


class CNNSE(nn.Module):
    'Convolutional Neural Networks with SEnet'
    
    def __init__(self, c_in=1, out_dim=128) -> None:
        super().__init__()
        self.body = torch.nn.ModuleList(
            [torch.nn.Conv3d(c_in, 64, 3, 1), torch.nn.MaxPool3d(2, 2), SENet(64)])
        for c in [64, 128, 256]:
            self.body.extend([
                torch.nn.Conv3d(c, c*2, 3, 1), 
                torch.nn.MaxPool3d(4, 2), 
                SENet(c*2)
            ])
        self.global_avg_pooling = lambda x:torch.mean(x, (1, 2, 3))
        self.linear_1 = torch.nn.Linear(512, 256)
        self.linear_2 = torch.nn.Linear(256, out_dim)
        
    def forward(self, images):
        x = images
        for layer in self.body:
            x = layer(x)
            print(x.size(-1))
        x = x.permute((0, 2, 3, 4, 1))
        x = self.global_avg_pooling(x)
        x = self.linear_2(self.linear_1(x)).unsqueeze(-1)

        return x


class MAFM(nn.Module):
    'Feature fusion and classification module'
    
    def __init__(self, image_dim, indicator_dim) -> None:
        super().__init__()
        self.transformer_layer1 = torch.nn.TransformerEncoderLayer(
            d_model=1, nhead=1, dim_feedforward=1)
        self.transformer_layer2 = torch.nn.TransformerEncoderLayer(
            d_model=1, nhead=1, dim_feedforward=1)
        self.linear = torch.nn.Linear(image_dim+indicator_dim, 3)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, image_embeds, indicator_embeds):
        indicator_embeds = indicator_embeds.transpose(1, 2)
        img = self.transformer_layer1(image_embeds)
        ind = self.transformer_layer2(indicator_embeds)
        x = torch.cat((img, ind), dim=1)
        x = x.squeeze(dim=-1)
        x = self.softmax(self.linear(x))

        return x


class AlzheimerModel(nn.Module):
    
    def __init__(self, indicator_dim) -> None:
        super().__init__()
        self.cnn_se = CNNSE(1, 128)
        self.mafm = MAFM(128, indicator_dim)

    def forward(self, images, indicators):
        ret = self.mafm(self.cnn_se(images), indicators)

        return ret


