import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STN(nn.Module):
    def __init__(self, input_shape, output_shape, device, theta_const=None):
        super(STN, self).__init__()
        self.theta = theta_const
        self.padding = (input_shape[-2]%2, input_shape[-1]%2)
        self.input_shape = input_shape
        self.output_shape = output_shape
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_shape[-3], 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, padding=self.padding),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, padding=self.padding),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, padding=self.padding)            
        )

        # Regressor for the 3 * 2 affine matrix
        #self.loc_input = self.calculate_input(input_shape)
        self.loc_input = int((input_shape[-1]/8)*(input_shape[-2]/8)*16)
        self.fc_loc = nn.Sequential(
            nn.Linear(self.loc_input, 20),
            nn.Linear(20, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[1].weight.data = torch.zeros(*self.fc_loc[1].weight.data.shape).to(device)
        self.fc_loc[1].bias.data = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).to(device)

    def calculate_input(self, input_dims):
        dims = torch.zeros(*input_dims)
        dims = self.localization(dims)
        return int(np.prod(dims.size()))

    # Spatial transformer network forward function
    def forward(self, x):
        if self.theta == None:
            xs = self.localization(x)
            xs = xs.view(-1, self.loc_input)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)
        else:
          theta = self.theta
        grid = F.affine_grid(theta, (x.shape[0], *self.output_shape[-3:]))
        x = F.grid_sample(x, grid)

        return x



		
class Encoder(nn.Module):
    def __init__(self, input_shape, u_depth, filters1):
        super(Encoder, self).__init__()
        self.u_depth = u_depth
        self.encoder_layers = nn.ModuleList()
        filters = input_shape[-3]
        filters_ = filters1
        for d in range(self.u_depth):
            layer = nn.Sequential(
                nn.Conv2d(filters, filters_, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(filters_),
                nn.Conv2d(filters_, filters_, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(filters_)
            )
            
            if d<(self.u_depth-1):
                layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layer.append(nn.Dropout2d(p=0.1))
            filters = filters_
            filters_ = (2**(d+1))*filters1
            self.encoder_layers.append(layer)

    def forward(self, x):
        encoder_layers = []
        t = x

        for d in range(self.u_depth):
            if d<(self.u_depth-1):
                t = self.encoder_layers[d][:-2](t)
                encoder_layers.append(t)
                t = self.encoder_layers[d][-2:](t)
            else:
                t = self.encoder_layers[d](t)
                encoder_layers.append(t)

        return encoder_layers
		



class Joiner(nn.Module):
    def __init__(self, input_shape, udepth, n_inputs, filters1, thetas, device):
        super(Joiner, self).__init__()
        self.n_inputs = n_inputs
        self.filters1 = filters1
        self.thetas = thetas
        self.joiner_outputs = []
        self.joiner_layers = nn.ModuleList()
        self.depth_stn_layers = nn.ModuleList()
        self.device = device
        
        shape = list(input_shape)
        for d in range(udepth):
            layer = []
            
            filters = (2**d)*self.filters1
            warped_maps = []
            stn_layers = nn.ModuleList()
            for i in range(self.n_inputs):
                shape[-3] = filters
                shape[-2] = input_shape[-2]//(2**d)
                shape[-1] = input_shape[-1]//(2**d)
                if self.thetas==None:
                    stn_layers.append(STN(tuple(shape), tuple(shape), self.device, self.thetas))
                else:
                    stn_layers.append(STN(tuple(shape), tuple(shape), self.device, self.thetas[i]))
            self.depth_stn_layers.append(stn_layers)
            layer.append(nn.Conv2d(filters*n_inputs, filters, kernel_size=3, padding = 1))
            layer.append(nn.ReLU(True))
            layer.append(nn.BatchNorm2d(filters))
            layer.append(nn.Conv2d(filters, filters, kernel_size=3, padding = 1))
            layer.append(nn.ReLU(True))
            layer.append(nn.BatchNorm2d(filters))
            self.joiner_layers.append(nn.Sequential(*layer))

    def forward(self, encoder_outputs):
        udepth = len(encoder_outputs[0])
	self.joiner_outputs = []
        for d in range(udepth):
            filters = (2**d)*self.filters1
            warped_maps = []
            for i in range(self.n_inputs):
                t = self.depth_stn_layers[d][i](encoder_outputs[i][d])
                warped_maps.append(t)
            t = torch.cat(warped_maps, dim=1) if self.n_inputs > 1 else warped_maps[0]
            t = self.joiner_layers[d](t)
            self.joiner_outputs.append(t)
        return self.joiner_outputs
		
		
		
		
class Decoder(nn.Module):
    def __init__(self, udepth, filters1, device):
        super(Decoder, self).__init__()
        self.udepth = udepth
        self.filters1 = filters1
        self.decoder_layers = nn.ModuleList()
        self.device = device
        
        for d in reversed(range(self.udepth-1)):
            filters = (2**d)*self.filters1
            layer = []
            layer.append(nn.ConvTranspose2d(filters*2, filters, kernel_size=3, stride=2, padding=1, output_padding=1))
            layer.append(nn.Dropout2d(p=0.1))
            layer.append(nn.Conv2d(filters*2, filters, kernel_size=3, padding=1))
            layer.append(nn.ReLU(True))
            layer.append(nn.BatchNorm2d(filters))
            layer.append(nn.Conv2d(filters, filters, kernel_size=3, padding=1))
            layer.append(nn.ReLU(True))
            layer.append(nn.BatchNorm2d(filters))
            self.decoder_layers.append(nn.Sequential(*layer))

    def forward(self, joiner_outputs):
        t = joiner_outputs[-1]
        for d in reversed(range(self.udepth-1)):
            filters = (2**d)*self.filters1
            t = self.decoder_layers[self.udepth-2-d][0](t)
            t = torch.cat((joiner_outputs[d], t), dim=1)
            t = self.decoder_layers[self.udepth-2-d][1:](t)
        return t
		
		
		
		
class UNetXST(nn.Module):
    def __init__(self, input_shape, n_inputs, n_output_channels, thetas, udepth, filters1, device):
        super(UNetXST, self).__init__()
        self.n_inputs = n_inputs
        self.input_shape = input_shape
        self.n_output_channels = n_output_channels
        self.thetas = thetas
        self.udepth = udepth
        self.filters1 = filters1

        self.encoder = nn.ModuleList([Encoder(self.input_shape, self.udepth, self.filters1) for i in range(n_inputs)])
        self.joiner = Joiner(self.input_shape, self.udepth, self.n_inputs, self.filters1, self.thetas, device)
        self.decoder = Decoder(self.udepth, self.filters1, device)

        self.prediction = nn.Sequential(
            nn.Conv2d(self.filters1, n_output_channels, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        encoder_outputs = []
        for i in range(self.n_inputs):
            encoder_outputs.append(self.encoder[i](inputs[i]))
        joiner_output = self.joiner(encoder_outputs)
        decoder_output = self.decoder(joiner_output)
        prediction = self.prediction(decoder_output)
        return prediction
