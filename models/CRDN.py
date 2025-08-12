import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['UNetRNN', 'VGG16RNN', 'ResNet18RNN', 'ResNet50RNN', 'ResNet34RNN', 'ResNet101RNN', 'ResNet152RNN', 'ResNet50UNet', 'ResNet50FCN']

class RDC(nn.Module):
    def __init__(self, hidden_dim, kernel_size, bias, decoder='GRU'):
        """
        Recurrent Decoding Cell (RDC) module.
        :param hidden_dim:
        :param kernel_size: conv kernel size
        :param bias: if or not to add a bias term
        :param decoder: <name> [options: 'vanilla, LSTM, GRU']
        """
        super(RDC, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias
        self.decoder = decoder
        self.gru_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, self.kernel_size,
                                     padding=self.padding, bias=self.bias)
        self.gru_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,
                                  padding=self.padding, bias=self.bias)
        self.lstm_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, self.kernel_size,
                                      padding=self.padding, bias=self.bias)
        self.vanilla_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,
                                      padding=self.padding, bias=self.bias)

    def forward(self, x_cur, h_pre, c_pre=None):
        if self.decoder == "LSTM":
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            c_pre_up = F.interpolate(c_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.lstm_catconv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_cur = f * c_pre_up + i * g
            h_cur = o * torch.tanh(c_cur)

            return h_cur, c_cur

        elif self.decoder == "GRU":
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)

            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.gru_catconv(combined)
            cc_r, cc_z = torch.split(combined_conv, self.hidden_dim, dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv(torch.cat([x_cur, r * h_pre_up], dim=1)))
            h_cur = z * h_pre_up + (1 - z) * h_hat

            return h_cur

        elif self.decoder == "vanilla":
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.vanilla_conv(combined)
            h_cur = torch.relu(combined_conv)

            return h_cur


class UNetRNN(nn.Module):
    def __init__(self, input_channel, n_classes, kernel_size, feature_scale=4, decoder="LSTM", bias=True):

        super(UNetRNN, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.input_channel, filters[0], is_batchnorm=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], is_batchnorm=True)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], is_batchnorm=True)

        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block2 = nn.Sequential(
            nn.Conv2d(filters[1], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block3 = nn.Sequential(
            nn.Conv2d(filters[2], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block4 = nn.Sequential(
            nn.Conv2d(filters[3], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block5 = nn.Sequential(
            nn.Conv2d(filters[4], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias, decoder=self.decoder)

    def forward(self, input, cell_state=None):
        conv1 = self.conv1(input)  # 1,filters[0]

        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)  # 1/2,filters[1]

        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)  # 1/4,filters[2]

        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)  # 1/8,filters[3]

        maxpool4 = self.maxpool4(conv4)
        conv5 = self.center(maxpool4)  # 1/16,filters[4]

        x1 = self.score_block5(conv5)  # 1/16,class
        x2 = self.score_block4(conv4)  # 1/8,class
        x3 = self.score_block3(conv3)  # 1/4,class
        x4 = self.score_block2(conv2)  # 1/2,class
        x5 = self.score_block1(conv1)  # 1,class

        h0 = self._init_cell_state(x1)  # 1/16,512

        # Decode
        if self.decoder == "LSTM":
            # init c0
            if cell_state is not None:
                raise NotImplementedError()
            else:
                c0 = self._init_cell_state(h0)

            h1, c1 = self.RDC(x_cur=x1, h_pre=h0, c_pre=c0)  # 1/16,class
            h2, c2 = self.RDC(x_cur=x2, h_pre=h1, c_pre=c1)  # 1/8,class
            h3, c3 = self.RDC(x_cur=x3, h_pre=h2, c_pre=c2)  # 1/4,class
            h4, c4 = self.RDC(x_cur=x4, h_pre=h3, c_pre=c3)  # 1/2,class
            h5, c5 = self.RDC(x_cur=x5, h_pre=h4, c_pre=c4)  # 1,class


        elif self.decoder == "GRU":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class

        elif self.decoder == "vanilla":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class

        else:
            raise NotImplementedError

        # return h1,h2,h3,h4,h5
        return h5

    def _init_cell_state(self, tensor):
        if torch.cuda.is_available():
            return torch.zeros(tensor.size()).cuda(0)
        else:
            return torch.zeros(tensor.size())


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        outputs1 = inputs1
        outputs2 = F.interpolate(outputs2, size=[outputs1.size(2), outputs1.size(3)], mode='bilinear',
                                 align_corners=True)

        return self.conv(torch.cat([outputs1, outputs2], 1))


# Simplified versions of other models for basic functionality
# For full implementation, refer to the original repository

def VGG16RNN(**kwargs):
    """Placeholder for VGG16RNN - not implemented in this simplified version"""
    raise NotImplementedError("VGG16RNN not implemented in this simplified version")

def ResNet18RNN(**kwargs):
    """Placeholder for ResNet18RNN - not implemented in this simplified version"""
    raise NotImplementedError("ResNet18RNN not implemented in this simplified version")

def ResNet34RNN(**kwargs):
    """Placeholder for ResNet34RNN - not implemented in this simplified version"""
    raise NotImplementedError("ResNet34RNN not implemented in this simplified version")

def ResNet50RNN(**kwargs):
    """Placeholder for ResNet50RNN - not implemented in this simplified version"""
    raise NotImplementedError("ResNet50RNN not implemented in this simplified version")

def ResNet101RNN(**kwargs):
    """Placeholder for ResNet101RNN - not implemented in this simplified version"""
    raise NotImplementedError("ResNet101RNN not implemented in this simplified version")

def ResNet152RNN(**kwargs):
    """Placeholder for ResNet152RNN - not implemented in this simplified version"""
    raise NotImplementedError("ResNet152RNN not implemented in this simplified version")

def ResNet50UNet(**kwargs):
    """Placeholder for ResNet50UNet - not implemented in this simplified version"""
    raise NotImplementedError("ResNet50UNet not implemented in this simplified version")

def ResNet50FCN(**kwargs):
    """Placeholder for ResNet50FCN - not implemented in this simplified version"""
    raise NotImplementedError("ResNet50FCN not implemented in this simplified version")
