# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import torch.nn as nn
import torch
from thop import profile

class Model(nn.Module):

    def __init__(self, para, instance_norm=False, instance_norm_level_1=False):
        super(Model, self).__init__()

        self.level = para.pynet_level
        self.n_colors = para.n_colors
        self.n_features = para.n_features

        if para.with_in:
            instance_norm = True
 
        # num of channels in U-Net
        nc = self.n_features

        # num of channels in level_x
        nf_l1 = para.n_features_level_1
        nf_l2 = nf_l1*2
        nf_l3 = nf_l2*2
        nf_l4 = nf_l3*2
        nf_l5 = nf_l4*2
        
        self.conv_l1_d1 = ConvMultiBlock(self.n_colors, nc, 3, instance_norm=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv_l2_d1 = ConvMultiBlock(nc, nc*2, 3, instance_norm=instance_norm)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv_l3_d1 = ConvMultiBlock(nc*2, nc*4, 3, instance_norm=instance_norm)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv_l4_d1 = ConvMultiBlock(nc*4, nc*8, 3, instance_norm=instance_norm)
        self.pool4 = nn.MaxPool2d(2, 2)

        # -------------------------------------

        self.conv_l5_d1 = ConvMultiBlock(nc*8, nf_l5, 3, instance_norm=instance_norm)
        self.conv_l5_d2 = ConvMultiBlock(nf_l5, nf_l5, 3, instance_norm=instance_norm)
        self.conv_l5_d3 = ConvMultiBlock(nf_l5, nf_l5, 3, instance_norm=instance_norm)
        self.conv_l5_d4 = ConvMultiBlock(nf_l5, nc*8, 3, instance_norm=instance_norm)

        self.conv_t4a = UpsampleConvLayer(nc*8, nf_l4, 3)
        self.conv_t4b = UpsampleConvLayer(nc*8, nf_l4, 3)

        self.conv_l5_out = ConvLayer(nc*8, 3, kernel_size=3, stride=1, relu=False)
        self.output_l5 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l4_d3 = ConvMultiBlock(nc*8+nf_l4, nf_l4, 3, instance_norm=instance_norm)
        self.conv_l4_d4 = ConvMultiBlock(nf_l4, nf_l4, 3, instance_norm=instance_norm)
        self.conv_l4_d5 = ConvMultiBlock(nf_l4, nf_l4, 3, instance_norm=instance_norm)
        self.conv_l4_d6 = ConvMultiBlock(nf_l4, nf_l4, 3, instance_norm=instance_norm)
        self.conv_l4_d8 = ConvMultiBlock(nf_l4+nf_l4, nc*8, 3, instance_norm=instance_norm)

        self.conv_t3a = UpsampleConvLayer(nc*8, nf_l3, 3)
        self.conv_t3b = UpsampleConvLayer(nc*8, nf_l3, 3)

        self.conv_l4_out = ConvLayer(nc*8, 3, kernel_size=3, stride=1, relu=False)
        self.output_l4 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l3_d3 = ConvMultiBlock(nc*4+nf_l3, nf_l3, 5, instance_norm=instance_norm)
        self.conv_l3_d4 = ConvMultiBlock(nf_l3*2, nf_l3, 5, instance_norm=instance_norm)
        self.conv_l3_d5 = ConvMultiBlock(nf_l3*2, nf_l3, 5, instance_norm=instance_norm)
        self.conv_l3_d6 = ConvMultiBlock(nf_l3*2, nf_l3, 5, instance_norm=instance_norm)
        self.conv_l3_d8 = ConvMultiBlock(nf_l3*2+nc*4+nf_l3, nc*4, 3, instance_norm=instance_norm)

        self.conv_t2a = UpsampleConvLayer(nc*4, nf_l2, 3)
        self.conv_t2b = UpsampleConvLayer(nc*4, nf_l2, 3)

        self.conv_l3_out = ConvLayer(nc*4, 3, kernel_size=3, stride=1, relu=False)
        self.output_l3 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l2_d3 = ConvMultiBlock(nc*2+nf_l2, nf_l2, 5, instance_norm=instance_norm)
        self.conv_l2_d5 = ConvMultiBlock(nc*2+nf_l2*2, nf_l2, 7, instance_norm=instance_norm)
        self.conv_l2_d6 = ConvMultiBlock(nf_l2*3, nf_l2, 7, instance_norm=instance_norm)
        self.conv_l2_d7 = ConvMultiBlock(nf_l2*3, nf_l2, 7, instance_norm=instance_norm)
        self.conv_l2_d8 = ConvMultiBlock(nf_l2*3, nf_l2, 7, instance_norm=instance_norm)
        self.conv_l2_d10 = ConvMultiBlock(nc*2+nf_l2*3, nf_l2, 5, instance_norm=instance_norm)
        self.conv_l2_d12 = ConvMultiBlock(nf_l2+nf_l2*2, nc*2, 3, instance_norm=instance_norm)

        self.conv_t1a = UpsampleConvLayer(nc*2, nf_l1, 3)
        self.conv_t1b = UpsampleConvLayer(nc*2, nf_l1, 3)

        self.conv_l2_out = ConvLayer(nc*2, 3, kernel_size=3, stride=1, relu=False)
        self.output_l2 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l1_d3 = ConvMultiBlock(nc+nf_l1,   nf_l1, 5, instance_norm=False)
        self.conv_l1_d5 = ConvMultiBlock(nc+nf_l1*2, nf_l1, 7, instance_norm=instance_norm_level_1)

        self.conv_l1_d6 = ConvMultiBlock(nf_l1*3, nf_l1, 9, instance_norm=instance_norm_level_1)
        self.conv_l1_d7 = ConvMultiBlock(nf_l1*4, nf_l1, 9, instance_norm=instance_norm_level_1)
        self.conv_l1_d8 = ConvMultiBlock(nf_l1*4, nf_l1, 9, instance_norm=instance_norm_level_1)
        self.conv_l1_d9 = ConvMultiBlock(nf_l1*4, nf_l1, 9, instance_norm=instance_norm_level_1)

        self.conv_l1_d10 = ConvMultiBlock(nf_l1*4, nf_l1, 7, instance_norm=instance_norm_level_1)
        self.conv_l1_d12 = ConvMultiBlock(nc+nf_l1*3, nf_l1, 5, instance_norm=instance_norm_level_1)
        self.conv_l1_d14 = ConvMultiBlock(nf_l1*2+nf_l1+nc, nc, 3, instance_norm=False)

        self.conv_l1_out = ConvLayer(nc, 3, kernel_size=3, stride=1, relu=False)
        self.output_l1 = nn.Sigmoid()

        self.conv_t0 = UpsampleConvLayer(nc, 16, 3)

        # -------------------------------------

        self.conv_l0_d1 = ConvLayer(3, 3, kernel_size=3, stride=1, relu=False)
        self.output_l0 = nn.Sigmoid()

    def level_5(self, pool4):

        conv_l5_d1 = self.conv_l5_d1(pool4)
        conv_l5_d2 = self.conv_l5_d2(conv_l5_d1) + conv_l5_d1
        conv_l5_d3 = self.conv_l5_d3(conv_l5_d2) + conv_l5_d2
        conv_l5_d4 = self.conv_l5_d4(conv_l5_d3)

        conv_t4a = self.conv_t4a(conv_l5_d4)
        conv_t4b = self.conv_t4b(conv_l5_d4)

        conv_l5_out = self.conv_l5_out(conv_l5_d4)
        output_l5 = self.output_l5(conv_l5_out)

        #return output_l5, conv_t4a, conv_t4b
        return conv_l5_out, conv_t4a, conv_t4b

    def level_4(self, conv_l4_d1, conv_t4a, conv_t4b):

        conv_l4_d2 = torch.cat([conv_l4_d1, conv_t4a], 1)

        conv_l4_d3 = self.conv_l4_d3(conv_l4_d2)
        conv_l4_d4 = self.conv_l4_d4(conv_l4_d3) + conv_l4_d3
        conv_l4_d5 = self.conv_l4_d5(conv_l4_d4) + conv_l4_d4
        conv_l4_d6 = self.conv_l4_d6(conv_l4_d5)

        conv_l4_d7 = torch.cat([conv_l4_d6, conv_t4b], 1)
        conv_l4_d8 = self.conv_l4_d8(conv_l4_d7)

        conv_t3a = self.conv_t3a(conv_l4_d8)
        conv_t3b = self.conv_t3b(conv_l4_d8)

        conv_l4_out = self.conv_l4_out(conv_l4_d8)
        output_l4 = self.output_l4(conv_l4_out)

        #return output_l4, conv_t3a, conv_t3b
        return conv_l4_out, conv_t3a, conv_t3b

    def level_3(self, conv_l3_d1, conv_t3a, conv_t3b):

        conv_l3_d2 = torch.cat([conv_l3_d1, conv_t3a], 1)

        conv_l3_d3 = self.conv_l3_d3(conv_l3_d2)
        conv_l3_d4 = self.conv_l3_d4(conv_l3_d3) + conv_l3_d3
        conv_l3_d5 = self.conv_l3_d5(conv_l3_d4) + conv_l3_d4
        conv_l3_d6 = self.conv_l3_d6(conv_l3_d5)

        conv_l3_d7 = torch.cat([conv_l3_d6, conv_l3_d1, conv_t3b], 1)
        conv_l3_d8 = self.conv_l3_d8(conv_l3_d7)

        conv_t2a = self.conv_t2a(conv_l3_d8)
        conv_t2b = self.conv_t2b(conv_l3_d8)

        conv_l3_out = self.conv_l3_out(conv_l3_d8)
        output_l3 = self.output_l3(conv_l3_out)

        #return output_l3, conv_t2a, conv_t2b
        return conv_l3_out, conv_t2a, conv_t2b

    def level_2(self, conv_l2_d1, conv_t2a, conv_t2b):

        conv_l2_d2 = torch.cat([conv_l2_d1, conv_t2a], 1)
        conv_l2_d3 = self.conv_l2_d3(conv_l2_d2)
        conv_l2_d4 = torch.cat([conv_l2_d3, conv_l2_d1], 1)

        conv_l2_d5 = self.conv_l2_d5(conv_l2_d4)
        conv_l2_d6 = self.conv_l2_d6(conv_l2_d5) + conv_l2_d5
        conv_l2_d7 = self.conv_l2_d7(conv_l2_d6) + conv_l2_d6
        conv_l2_d8 = self.conv_l2_d8(conv_l2_d7)
        conv_l2_d9 = torch.cat([conv_l2_d8, conv_l2_d1], 1)

        conv_l2_d10 = self.conv_l2_d10(conv_l2_d9)
        conv_l2_d11 = torch.cat([conv_l2_d10, conv_t2b], 1)
        conv_l2_d12 = self.conv_l2_d12(conv_l2_d11)

        conv_t1a = self.conv_t1a(conv_l2_d12)
        conv_t1b = self.conv_t1b(conv_l2_d12)

        conv_l2_out = self.conv_l2_out(conv_l2_d12)
        output_l2 = self.output_l2(conv_l2_out)

        #return output_l2, conv_t1a, conv_t1b
        return conv_l2_out, conv_t1a, conv_t1b

    def level_1(self, conv_l1_d1, conv_t1a, conv_t1b):

        conv_l1_d2 = torch.cat([conv_l1_d1, conv_t1a], 1)
        conv_l1_d3 = self.conv_l1_d3(conv_l1_d2)        
        conv_l1_d4 = torch.cat([conv_l1_d3, conv_l1_d1], 1)
        #print('\t>>conv_l1_d3: ', conv_l1_d3.shape)
        #print('\t>>conv_l1_d1: ', conv_l1_d1.shape)

        #print('\nbegin conv_l1_d5 ..., with input conv_l1_d4.')
        #print('\tconv_l1_d4: ', conv_l1_d4.shape)
        conv_l1_d5 = self.conv_l1_d5(conv_l1_d4)
        #print('\t>>conv_l1_d5: ', conv_l1_d5.shape)

        conv_l1_d6 = self.conv_l1_d6(conv_l1_d5)
        conv_l1_d7 = self.conv_l1_d7(conv_l1_d6) + conv_l1_d6
        conv_l1_d8 = self.conv_l1_d8(conv_l1_d7) + conv_l1_d7
        conv_l1_d9 = self.conv_l1_d9(conv_l1_d8) + conv_l1_d8

        conv_l1_d10 = self.conv_l1_d10(conv_l1_d9)
        conv_l1_d11 = torch.cat([conv_l1_d10, conv_l1_d1], 1)

        conv_l1_d12 = self.conv_l1_d12(conv_l1_d11)
        conv_l1_d13 = torch.cat([conv_l1_d12, conv_t1b, conv_l1_d1], 1)

        conv_l1_d14 = self.conv_l1_d14(conv_l1_d13)
        conv_t0 = self.conv_t0(conv_l1_d14)

        conv_l1_out = self.conv_l1_out(conv_l1_d14)
        output_l1 = self.output_l1(conv_l1_out)

        #return output_l1, conv_t0
        return conv_l1_out, conv_t0

    def level_0(self, output_l1):

        conv_l0_d1 = self.conv_l0_d1(output_l1)
        output_l0 = self.output_l0(conv_l0_d1)

        #return output_l0
        return conv_l0_d1

    def forward(self, x, profile_flag=False):
        if profile_flag:
            return self.profile_forward(x)

        conv_l1_d1 = self.conv_l1_d1(x)
        pool1 = self.pool1(conv_l1_d1)

        conv_l2_d1 = self.conv_l2_d1(pool1)
        pool2 = self.pool2(conv_l2_d1)

        conv_l3_d1 = self.conv_l3_d1(pool2)
        pool3 = self.pool3(conv_l3_d1)

        conv_l4_d1 = self.conv_l4_d1(pool3)
        pool4 = self.pool4(conv_l4_d1)

        output_l5, conv_t4a, conv_t4b = self.level_5(pool4)

        if self.level < 5:
            output_l4, conv_t3a, conv_t3b = self.level_4(conv_l4_d1, conv_t4a, conv_t4b)
        if self.level < 4:
            output_l3, conv_t2a, conv_t2b = self.level_3(conv_l3_d1, conv_t3a, conv_t3b)
        if self.level < 3:
            output_l2, conv_t1a, conv_t1b = self.level_2(conv_l2_d1, conv_t2a, conv_t2b)
        if self.level < 2:
            output_l1, conv_t0 = self.level_1(conv_l1_d1, conv_t1a, conv_t1b)
        if self.level < 1:
            output_l0 = self.level_0(output_l1)

        if self.level == 0:
            enhanced = x[:,:3,:,:] + output_l0
        if self.level == 1:
            enhanced = output_l1
        if self.level == 2:
            enhanced = output_l2
        if self.level == 3:
            enhanced = output_l3
        if self.level == 4:
            enhanced = output_l4
        if self.level == 5:
            enhanced = output_l5

        return output_l0, enhanced

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    # For calculating GMACs
    def profile_forward(self, x):
        outputs, hs = [], []
        batch_size, channels, height, width = x.shape
        #print('x: ', x.shape)
        #s = torch.zeros(batch_size, self.n_colors, height, width).to(self.device)
        conv_l1_d1 = self.conv_l1_d1(x)
        pool1 = self.pool1(conv_l1_d1)

        conv_l2_d1 = self.conv_l2_d1(pool1)
        pool2 = self.pool2(conv_l2_d1)

        conv_l3_d1 = self.conv_l3_d1(pool2)
        pool3 = self.pool3(conv_l3_d1)

        conv_l4_d1 = self.conv_l4_d1(pool3)
        pool4 = self.pool4(conv_l4_d1)

        output_l5, conv_t4a, conv_t4b = self.level_5(pool4)

        if self.level < 5:
            output_l4, conv_t3a, conv_t3b = self.level_4(conv_l4_d1, conv_t4a, conv_t4b)
        if self.level < 4:
            output_l3, conv_t2a, conv_t2b = self.level_3(conv_l3_d1, conv_t3a, conv_t3b)
        if self.level < 3:
            output_l2, conv_t1a, conv_t1b = self.level_2(conv_l2_d1, conv_t2a, conv_t2b)
        if self.level < 2:
            output_l1, conv_t0 = self.level_1(conv_l1_d1, conv_t1a, conv_t1b)
        if self.level < 1:
            output_l0 = self.level_0(output_l1)

        if self.level == 0:
            enhanced = output_l0
        if self.level == 1:
            enhanced = output_l1
        if self.level == 2:
            enhanced = output_l2
        if self.level == 3:
            enhanced = output_l3
        if self.level == 4:
            enhanced = output_l4
        if self.level == 5:
            enhanced = output_l5

        return enhanced,enhanced


class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, max_conv_size, instance_norm):

        super(ConvMultiBlock, self).__init__()
        self.max_conv_size = max_conv_size

        self.conv_3a = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)
        self.conv_3b = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 5:
            self.conv_5a = ConvLayer(in_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)
            self.conv_5b = ConvLayer(out_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 7:
            self.conv_7a = ConvLayer(in_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)
            self.conv_7b = ConvLayer(out_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 9:
            self.conv_9a = ConvLayer(in_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)
            self.conv_9b = ConvLayer(out_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)

    def forward(self, x):

        out_3 = self.conv_3a(x)
        output_tensor = self.conv_3b(out_3)
        #print('\t-> x: ', x.shape)
        #print('\t-> output_tensor : ', output_tensor.shape)

        if self.max_conv_size >= 5:
            out_5 = self.conv_5a(x)
            out_5 = self.conv_5b(out_5)
            output_tensor = torch.cat([output_tensor, out_5], 1)
            #print('\t-> out_5: ', out_5.shape)
            #print('\t-> output_tensor : ', output_tensor.shape)

        if self.max_conv_size >= 7:
            out_7 = self.conv_7a(x)
            out_7 = self.conv_7b(out_7)
            output_tensor = torch.cat([output_tensor, out_7], 1)
            #print('\t-> out_7 : ', out_7.shape)
            #print('\t-> output_tensor : ', output_tensor.shape)

        if self.max_conv_size >= 9:
            out_9 = self.conv_9a(x)
            out_9 = self.conv_9b(out_9)
            output_tensor = torch.cat([output_tensor, out_9], 1)
            #print('\t-> out_9 : ', out_9.shape)
            #print('\t-> output_tensor : ', output_tensor.shape) 

        return output_tensor


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=2, stride=1, relu=True):

        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.upsample(x)
        out = self.reflection_pad(out)
        out = self.conv2d(out)

        if self.relu:
            out = self.relu(out)

        return out
 

def feed(model, iter_samples):
    inputs = iter_samples[0]
    #print('>>inputs: ', inputs.shape)
    outputs = model(inputs)

    return outputs

def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, 8, H, W).cuda()
    profile_flag = True 

    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops, params
