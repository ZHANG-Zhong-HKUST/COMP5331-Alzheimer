from statistics import LinearRegression
from turtle import forward
import torch


class SENet(nn.Module):
    'Squeeze-and-Excitation networks'

    def __init__(self, n_channels, dim_red_factor=16) -> None:
        super().__init__()
        self.global_avg_pooling = lambda x:torch.mean(x, (0, 1, 2), keepdim=True)
        self.linear_1 = torch.nn.Linear(n_channels, n_channels/dim_red_factor)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(n_channels/dim_red_factor, n_channels)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, inputs):
        x = self.global_avg_pooling(inputs)
        x = self.linear_2(self.relu(self.linear_1(x)))
        attns = self.sigmoid(x)
        ret = torch.mul(inputs, attns)
        
        return ret


class CNNSE(nn.Module):
    'Convolutional Neural Networks with SEnet'
    
    def __init__(self, in_channels=1, out_dim=128) -> None:
        super().__init__()
        self.body = torch.nn.ModuleList(
            [torch.nn.Conv3d(in_channels, 64, 4, 2), torch.nn.MaxPool3d(4, 2), SENet(64)])
        for c in [64, 128, 256]:
            self.body.extend([
                torch.nn.Conv3d(c, c*2, 4, 2), 
                torch.nn.MaxPool3d(4, 2), 
                SENet(c*2)
            ])
        self.global_avg_pooling = lambda x:torch.mean(x, (0, 1, 2)).unsqueeze(-1)
        self.linear_1 = torch.nn.Linear(512, 256)
        self.linear_2 = torch.nn.Linear(256, out_dim)
        
    def forward(self, images):
        x = images
        for layer in self.body:
            x = layer(x)
        x = self.linear_2(self.linear_1(self.global_avg_pooling(x)))

        return x


class MAFM(nn.Module):
    'Feature fusion and classification module'
    
    def __init__(self, image_dim, indicator_dim) -> None:
        super().__init__()
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=1, nhead=8, dim_feedforward=1)
        self.linear = torch.nn.Linear(image_dim+indicator_dim, 3)
        self.softmax = torch.nn.Softmax()
    
    def forward(self, image_embeds, indicator_embeds):

        img = self.transformer_layer(image_embeds)
        ind = self.transformer_layer(indicator_embeds)
        x = torch.cat((img, ind), dim=0)
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


