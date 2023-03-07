import numpy as np

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

import cv2
import kornia.filters.sobel as sobel_filter

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class SubDecoder(nn.Module):
    def __init__(self, in_channels, act_fn):
        super(SubDecoder, self).__init__()

        self.conv_block = conv_block(in_channels, in_channels, act_fn)

        self.region_output = nn.Conv2d(in_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.edge_output = nn.Conv2d(in_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.distance_output = nn.Conv2d(in_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x, pre_result=None):
        out = self.conv_block(x)

        region_out = self.region_output(out)
        edge_output = self.edge_output(out)
        distance_output = self.distance_output(out)

        if pre_result is None:
            return region_out, edge_output, distance_output
        else:
            return region_out + F.interpolate(pre_result[0], scale_factor=2, mode='bilinear'), \
                   edge_output + F.interpolate(pre_result[1], scale_factor=2, mode='bilinear'), \
                   distance_output + F.interpolate(pre_result[2], scale_factor=2, mode='bilinear')

class ASPPModule(nn.Module) :
    def __init__(self, in_channels, out_channels, act_fn):
        super(ASPPModule, self).__init__()

        self.stem_conv = conv_block_2(in_channels * 3, in_channels, act_fn)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv3x3_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv3x3_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv3x3_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(18, 18), dilation=(18, 18)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

        self.out_conv1 = conv_block(out_channels, out_channels, act_fn)
        self.global_map_output = nn.Conv2d(out_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.out_conv2 = conv_block(out_channels * 2, out_channels, act_fn)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, low, full, high):
        x = torch.cat([low, full, high], dim=1)

        stem_x = self.stem_conv(x)

        branch1 = self.conv1x1(stem_x)
        branch2 = self.conv3x3_6(stem_x)
        branch3 = self.conv3x3_12(stem_x)
        branch4 = self.conv3x3_18(stem_x)

        aspp = branch1 + branch2 + branch3 + branch4
        output = self.out_conv1(aspp)
        global_map = self.global_map_output(output)
        reverse_global_map = 1 - torch.sigmoid(global_map)

        output = self.out_conv2(torch.cat([(output * torch.sigmoid(global_map)) * self.alpha, (output * reverse_global_map) * self.beta], dim=1))

        return output, global_map

class M3FPolypSegNet(nn.Module):
    def __init__(self, device, num_channels, image_size, init_features, power_ratio=0.5, edge_threshold=0.5, upsample_method='bilinear'):
        super(M3FPolypSegNet, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss()

        self.device = device
        self.in_dim = num_channels
        self.n_class = 1
        self.num_filters = init_features
        self.image_size = image_size
        self.power_ratio = power_ratio
        self.edge_threshold = edge_threshold
        self.low_freq_mask, self.high_freq_mask = self.decompose(image_size, 10)

        act_fn = nn.ReLU(inplace=True)
        self.pool = maxpool()

        # Full-Frequency Stream Encoder
        self.full_down1 = conv_block_2(self.in_dim, self.num_filters * 1, act_fn)
        self.full_down2 = conv_block_2(self.num_filters * 1, self.num_filters * 2, act_fn)
        self.full_down3 = conv_block_2(self.num_filters * 2, self.num_filters * 4, act_fn)
        self.full_down4 = conv_block_2(self.num_filters * 4, self.num_filters * 8, act_fn)

        # Low-Frequency Stream Encoder
        self.low_down1 = conv_block_2(self.in_dim, self.num_filters * 1, act_fn)
        self.low_down2 = conv_block_2(self.num_filters * 1, self.num_filters * 2, act_fn)
        self.low_down3 = conv_block_2(self.num_filters * 2, self.num_filters * 4, act_fn)
        self.low_down4 = conv_block_2(self.num_filters * 4, self.num_filters * 8, act_fn)

        # High-Frequency Stream Encoder
        self.high_down1 = conv_block_2(self.in_dim, self.num_filters * 1, act_fn)
        self.high_down2 = conv_block_2(self.num_filters * 1, self.num_filters * 2, act_fn)
        self.high_down3 = conv_block_2(self.num_filters * 2, self.num_filters * 4, act_fn)
        self.high_down4 = conv_block_2(self.num_filters * 4, self.num_filters * 8, act_fn)

        # Guide Conv
        self.guide_conv_full_down1 = conv_block_2(self.num_filters * 1, self.num_filters * 1, act_fn)
        self.guide_conv_full_down2 = conv_block_2(self.num_filters * 2, self.num_filters * 2, act_fn)
        self.guide_conv_full_down3 = conv_block_2(self.num_filters * 4, self.num_filters * 4, act_fn)
        self.guide_conv_full_down4 = conv_block_2(self.num_filters * 8, self.num_filters * 8, act_fn)

        # Frequency-Fusion Bridge
        self.bridge = ASPPModule(self.num_filters * 8, self.num_filters * 16, act_fn)
        # self.bridge = conv_block_2(self.num_filters * 8 * 3, self.num_filters * 16, act_fn)

        self.inter_reduce_cat1 = nn.Conv2d(self.num_filters * 8 * 3, self.num_filters * 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.inter_reduce_cat2 = nn.Conv2d(self.num_filters * 4 * 3, self.num_filters * 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.inter_reduce_cat3 = nn.Conv2d(self.num_filters * 2 * 3, self.num_filters * 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.inter_reduce_cat4 = nn.Conv2d(self.num_filters * 1 * 3, self.num_filters * 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Inter-Frequency Decoder
        self.inter_up1 = upsample(self.num_filters * 16, self.num_filters * 8, act_fn, upsample_method)
        self.inter_conv1 = conv_block(self.num_filters * 16, self.num_filters * 8, act_fn)

        self.inter_up2 = upsample(self.num_filters * 8, self.num_filters * 4, act_fn, upsample_method)
        self.inter_conv2 = conv_block(self.num_filters * 8, self.num_filters * 4, act_fn)

        self.inter_up3 = upsample(self.num_filters * 4, self.num_filters * 2, act_fn, upsample_method)
        self.inter_conv3 = conv_block(self.num_filters * 4, self.num_filters * 2, act_fn)

        self.inter_up4 = upsample(self.num_filters * 2, self.num_filters * 1, act_fn, upsample_method)
        self.inter_conv4 = conv_block(self.num_filters * 2, self.num_filters * 1, act_fn)

        # Sub-Decoder
        self.stage1_subdecoder = SubDecoder(self.num_filters * 8, act_fn)
        self.stage2_subdecoder = SubDecoder(self.num_filters * 4, act_fn)
        self.stage3_subdecoder = SubDecoder(self.num_filters * 2, act_fn)
        self.stage4_subdecoder = SubDecoder(self.num_filters * 1, act_fn)

    def forward(self, x):
        x_fft = fft.fftshift(fft.fft2(x))
        low_x = torch.real(fft.ifft2(fft.ifftshift(x_fft * self.low_freq_mask.to(x.get_device()))))
        high_x = torch.real(fft.ifft2(fft.ifftshift(x_fft * self.high_freq_mask.to(x.get_device()))))

        # self.plot_image(x_fft, x, low_x, high_x)

        # Full-Frequency Stream Encoder
        full_down1 = self.full_down1(x)  # [?, 3, 256, 256] -> [?, 32, 256, 256]
        full_pool1 = self.pool(full_down1)  # [?, 32, 256, 256] -> [?, 32, 128, 128]
        full_down2 = self.full_down2(full_pool1)  # [?, 32, 128, 128] -> [?, 64, 128, 128]
        full_pool2 = self.pool(full_down2)  # [?, 64, 128, 128] -> [?, 64, 64, 64]
        full_down3 = self.full_down3(full_pool2)  # [?, 64, 64, 64] -> [?, 128, 64, 64]
        full_pool3 = self.pool(full_down3)  # [?, 128, 64, 64] -> [?, 128, 32, 32]
        full_down4 = self.full_down4(full_pool3)  # [?, 128, 32, 32] -> [?, 256, 32, 32]
        full_pool4 = self.pool(full_down4)  # [?, 256, 32, 32] -> [?, 256, 16, 16]

        # Low-Frequency Stream Encoder
        low_down1 = self.low_down1(low_x) + self.guide_conv_full_down1(full_down1)  # [?, 3, 256, 256] -> [?, 32, 256, 256]
        low_pool1 = self.pool(low_down1)  # [?, 32, 256, 256] -> [?, 32, 128, 128]
        low_down2 = self.low_down2(low_pool1) + self.guide_conv_full_down2(full_down2)  # [?, 32, 128, 128] -> [?, 64, 128, 128]
        low_pool2 = self.pool(low_down2)  # [?, 64, 128, 128] -> [?, 64, 64, 64]
        low_down3 = self.low_down3(low_pool2) + self.guide_conv_full_down3(full_down3)  # [?, 64, 64, 64] -> [?, 128, 64, 64]
        low_pool3 = self.pool(low_down3)  # [?, 128, 64, 64] -> [?, 128, 32, 32]
        low_down4 = self.low_down4(low_pool3) + self.guide_conv_full_down4(full_down4)  # [?, 128, 32, 32] -> [?, 256, 32, 32]
        low_pool4 = self.pool(low_down4)  # [?, 256, 32, 32] -> [?, 256, 16, 16]

        # High-Frequency Stream Encoder
        high_down1 = self.high_down1(high_x) + self.guide_conv_full_down1(full_down1)  # [?, 3, 256, 256] -> [?, 32, 256, 256]
        high_pool1 = self.pool(high_down1)  # [?, 32, 256, 256] -> [?, 32, 128, 128]
        high_down2 = self.high_down2(high_pool1) + self.guide_conv_full_down2(full_down2)  # [?, 32, 128, 128] -> [?, 64, 128, 128]
        high_pool2 = self.pool(high_down2)  # [?, 64, 128, 128] -> [?, 64, 64, 64]
        high_down3 = self.high_down3(high_pool2) + self.guide_conv_full_down3(full_down3)  # [?, 64, 64, 64] -> [?, 128, 64, 64]
        high_pool3 = self.pool(high_down3)  # [?, 128, 64, 64] -> [?, 128, 32, 32]
        high_down4 = self.high_down4(high_pool3) + self.guide_conv_full_down4(full_down4)  # [?, 128, 32, 32] -> [?, 256, 32, 32]
        high_pool4 = self.pool(high_down4)  # [?, 256, 32, 32] -> [?, 256, 16, 16]

        # Frequency-Fusion Bridge
        frequency_agg, global_map0 = self.bridge(full_pool4, low_pool4, high_pool4)
        # frequency_agg = torch.cat([full_pool4, low_pool4, high_pool4], dim=1) # [?, 256, 16, 16] -> [?, 768, 16, 16]
        # frequency_agg = self.bridge(frequency_agg)                            # [?, 768, 16, 16] -> [?, 512, 16, 16]

        frequency_cat1 = self.inter_reduce_cat1(torch.cat([full_down4, low_down4, high_down4], dim=1))
        frequency_cat2 = self.inter_reduce_cat2(torch.cat([full_down3, low_down3, high_down3], dim=1))
        frequency_cat3 = self.inter_reduce_cat3(torch.cat([full_down2, low_down2, high_down2], dim=1))
        frequency_cat4 = self.inter_reduce_cat4(torch.cat([full_down1, low_down1, high_down1], dim=1))

        # Inter-Frequency Decoder
        inter_up1 = self.inter_up1(frequency_agg)  # [?, 512, 16, 16] -> [?, 256, 32, 32]
        inter_cat1 = torch.cat([inter_up1, frequency_cat1], dim=1)  # [?, 256, 32, 32] + [?, 256, 32, 32] -> [?, 512, 32, 32]
        inter_conv1 = self.inter_conv1(inter_cat1)  # [?, 512, 32, 32] -> [?, 256, 32, 32]

        region_output1, edge_output1, distance_output1 = self.stage1_subdecoder(inter_conv1)  # [?, 32, 256, 256] -> [?, 1, 256, 256]

        inter_up2 = self.inter_up2(inter_conv1)  # [?, 256, 32, 32] -> [?, 128, 64, 64]
        inter_cat2 = torch.cat([inter_up2, frequency_cat2], dim=1)  # [?, 128, 64, 64] + [?, 128, 64, 64] -> [?, 256, 64, 64]
        inter_conv2 = self.inter_conv2(inter_cat2)  # [?, 256, 64, 64] -> [?, 128, 64, 64]

        region_output2, edge_output2, distance_output2 = self.stage2_subdecoder(inter_conv2, pre_result=[region_output1, edge_output1, distance_output1])  # [?, 32, 256, 256] -> [?, 1, 256, 256]

        inter_up3 = self.inter_up3(inter_conv2)  # [?, 128, 64, 64] -> [?, 64, 128, 128]
        inter_cat3 = torch.cat([inter_up3, frequency_cat3], dim=1)  # [?, 64, 128, 128] + [?, 64, 128, 128] -> [?, 128, 128, 128]
        inter_conv3 = self.inter_conv3(inter_cat3)  # [?, 128, 128, 128] -> [?, 64, 128, 128]

        region_output3, edge_output3, distance_output3 = self.stage3_subdecoder(inter_conv3, pre_result=[region_output2, edge_output2, distance_output2])  # [?, 32, 256, 256] -> [?, 1, 256, 256]

        inter_up4 = self.inter_up4(inter_conv3)  # [?, 64, 128, 128] -> [?, 32, 256, 256]
        inter_cat4 = torch.cat([inter_up4, frequency_cat4], dim=1)  # [?, 32, 256, 256] + [?, 32, 256, 256] -> [?, 64, 256, 256]
        inter_conv4 = self.inter_conv4(inter_cat4)  # [?, 64, 256, 256] -> [?, 32, 256, 256]

        global_map0 = F.interpolate(global_map0, scale_factor=16, mode='bilinear')  # [?, 32, 256, 256] -> [?, 1, 256, 256]
        region_output4, edge_output4, distance_output4 = self.stage4_subdecoder(inter_conv4, pre_result=[region_output3, edge_output3, distance_output3])  # [?, 32, 256, 256] -> [?, 1, 256, 256]

        return [region_output4, edge_output4, distance_output4], \
               [region_output3, edge_output3, distance_output3], \
               [region_output2, edge_output2, distance_output2], \
               [region_output1, edge_output1, distance_output1], \
               global_map0

    def _calculate_criterion(self, y_pred, y_true):
        edge_true = sobel_filter(y_true)
        edge_true[edge_true >= 0.5] = 1; edge_true[edge_true < 0.5] = 0

        distance_true = torch.zeros_like(edge_true)
        for i, y_true_ in enumerate(y_true):
            y_true_ = np.array(np.transpose(y_true_.cpu().detach().numpy(), (1, 2, 0)) * 255, dtype=np.uint8)
            distance = cv2.distanceTransform(y_true_, cv2.DIST_L2, 5)
            cv2.normalize(distance, distance, 0, 1, cv2.NORM_MINMAX)
            distance_true[i] = torch.from_numpy(distance)

        region_loss, edge_loss, distance_loss = 0., 0., 0.

        for y_pred_, scale in zip(reversed(y_pred[:4]), [8, 4, 2, 1]):
            region_pred, edge_pred, distance_pred = y_pred_

            region_pred = F.interpolate(region_pred, scale_factor=scale, mode='bilinear')
            edge_pred = F.interpolate(edge_pred, scale_factor=scale, mode='bilinear')
            distance_pred = F.interpolate(distance_pred, scale_factor=scale, mode='bilinear')

            region_loss += F.binary_cross_entropy_with_logits(region_pred, y_true)  # * scale
            edge_loss += F.binary_cross_entropy_with_logits(edge_pred, edge_true)  # * scale
            distance_loss += F.mse_loss(torch.sigmoid(distance_pred), distance_true)  # * scale

        global_map_loss = self.criterion(y_pred[4], y_true)

        loss = region_loss + edge_loss + distance_loss + global_map_loss

        return loss

    def decompose(self, image_size, radius):
        low_freq_mask = torch.zeros((image_size, image_size))

        x_range = np.arange(0, image_size) - int(image_size / 2)
        y_range = np.arange(0, image_size) - int(image_size / 2)

        x_ms, y_ms = np.meshgrid(x_range, y_range)

        R = np.sqrt(x_ms ** 2 + y_ms ** 2)

        idxx, idxy = np.where(R <= radius)
        low_freq_mask[idxx, idxy] = 1
        high_freq_mask = 1 - low_freq_mask

        return low_freq_mask.to(self.device), high_freq_mask.to(self.device)

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean()

def conv_block_2(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim)
    )

    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

    return pool


def upsample(in_dim, out_dim, act_fn, upsample_method):
    if upsample_method == 'transpose':
        model = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=(2, 2), stride=(2, 2)),
            conv_block(out_dim, out_dim, act_fn)
        )
    else:
        model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv_block(in_dim, out_dim, act_fn)
        )

    return model


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim),
        act_fn
    )

    return model


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("You are using \"{}\" device.".format(device))
    # device, num_channels, image_size, init_features, power_ratio, edge_threshold
    model = FDNet(device, num_channels=3, image_size=256, power_ratio=0.5, init_features=32, edge_threshold=0.5).to(device)
    # model = Res2Net(Bottle2Neck, [3, 4, 6, 3], baseWidth=26, scale=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameter : {}".format(total_params))