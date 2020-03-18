import torch
import torch.nn as nn
from torch.nn import functional as F

class U_Net (nn.Module):

    def __init__ (self, down_layer_input_size, output_size, number_downsampling = 4, list_channels = [512] * 4, list_number_convs = [1] * 4, norm=False, dropout=0):

        super (U_Net, self).__init__ ()
        self.output_size = output_size
        self.number_downsampling = number_downsampling
        self.list_channels = list_channels
        self.list_number_convs = list_number_convs
        self.dropout = dropout
        self.down_layer_input_size = down_layer_input_size
        self.down_id_dense = nn.ModuleList()
        self.up_id_dense = nn.ModuleList()

        self.pre_layers = nn.ModuleList() 
        self.down_layers = nn.ModuleList()       
        self.up_layers = nn.ModuleList()

            
        for i in range (self.number_downsampling):

            down_layer = nn.ModuleList()

            for j in range (self.list_number_convs[i]):
                if i == 0 and j == 0:
                    layer = nn.Conv1d (self.down_layer_input_size, 2 * self.list_channels[i], 3, padding=1)
                elif j == 0:
                    layer = nn.Conv1d (2 * self.list_channels[i - 1], 2 * self.list_channels[i], 3, padding=1)
                else:
                    layer = nn.Conv1d (2 * self.list_channels[i], 2 * self.list_channels[i], 3, padding=1)
                down_layer.append (layer)


            if norm:
                continue

            down_layer.append (nn.AvgPool1d (kernel_size = 2, stride=2))

            self.down_layers.append(down_layer)

        for i in reversed (range (self.number_downsampling)):

            up_layer = nn.ModuleList()

            layer = nn.ConvTranspose2d (self.list_channels[i], self.list_channels[i], (2, 1), stride=(2, 1))
            up_layer.append (layer)


            for j in range (self.list_number_convs[i]):

                layer = nn.Conv1d (self.list_channels[i], 2 * self.list_channels[i], 3, padding=1)
                up_layer.append (layer)


            if norm:
                continue

            self.up_layers.append (up_layer)


        self.final_layer = nn.Conv1d (512, 512, 3, padding=1)
        self.final_dense = nn.Linear (512, self.output_size)
            
            
    def forward (self, x):

        x  = x.transpose(1, 2)

        down_outs = []

        for i in range (self.number_downsampling):

            for j in range (len (self.down_layers[i]) - 1):

                x = self.down_layers[i][j] (x)

            x1, x2 = torch.split (x, split_size_or_sections=int(x.shape[1]/2), dim=1)

            x = torch.sigmoid (x1) * torch.tanh (x2)
	    
            down_outs.append (x)

            x = self.down_layers[i][-1] (x)


        for i in range (self.number_downsampling):

            x = x.unsqueeze (3)

            x = self.up_layers[i][0] (x)
            
            x = torch.squeeze(x, dim = 3)

            x1, x2 = torch.split (x, split_size_or_sections=int(x.shape[1]/2), dim=1)

            x = x + down_outs[self.number_downsampling - i - 1]

            if self.dropout > 0:

                x = F.dropout(x, p=self.dropout)                

            for j in range (1, len (self.up_layers[i])):

                x = self.up_layers[i][j] (x)

                x1, x2 = torch.split (x, split_size_or_sections=int(x.shape[1]/2), dim=1)

                x = torch.sigmoid (x1) * torch.tanh (x2)

        x = torch.tanh(self.final_layer (x))
	    
        x = x.transpose (1, 2)
        return self.final_dense (x)
