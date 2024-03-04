import torch
from torch import nn


class PLE(nn.Module):
    def __init__(self, in_ch, in_sp, spa_ratio=8):
        super(PLE, self).__init__()
        self.in_ch = in_ch
        self.in_sp = in_sp
        self.inter_sp = in_sp
        self.inter_ch = in_ch // spa_ratio
        self.gx_spatial = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.inter_ch,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.inter_ch),
            nn.ReLU()
        )
        self.theta_spatial = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.inter_ch,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.inter_ch),
            nn.ReLU()
        )
        self.phi_spatial = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.inter_ch,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.inter_ch),
            nn.ReLU()

        )
        self.gg_spatial = nn.Sequential(
            nn.Conv3d(in_channels=self.in_sp * 2, out_channels=self.inter_sp,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.inter_sp),
            nn.ReLU()
        )

    def forward(self, R1, R2, debug=False):
        b, c, t, h, w = R1.size()
        theta_xs = self.theta_spatial(R1)  # 降低通道数1
        phi_xs = self.phi_spatial(R2)  # 降低通道数2
        theta_xs = theta_xs.view(b, self.inter_ch, t, -1)  # b, c,t, h, w->b, c,t, h*w
        theta_xs = theta_xs.permute(0, 2, 3, 1)  # 将b, c,t, h*w -> b,t,h*w, c  # 转置矩阵，也就是T
        phi_xs = phi_xs.view(b, self.inter_ch, t, -1)
        phi_xs = phi_xs.permute(0, 2, 1, 3)
        Gs = torch.matmul(theta_xs, phi_xs)  # 计算相关系数

        Gs_in = Gs.permute(0, 3, 1, 2).view(b, h * w, t, h, w)
        Gs_out = Gs.permute(0, 2, 1, 3).view(b, h * w, t, h, w)
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)  # 拼接相关矩阵
        Gs_joint = self.gg_spatial(Gs_joint)  # 通道降维
        return Gs_joint


class PLE_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, s_ratio=8, d_ratio=8):
        super(PLE_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial
        # self.use_spatial = use_spatial
        self.PLE = PLE(self.in_channel, self.in_spatial // 4, spa_ratio=s_ratio)
        num_channel_s = 3 * self.in_spatial // 4
        self.W_spatial = nn.Sequential(
            nn.Conv3d(in_channels=num_channel_s, out_channels=num_channel_s // d_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(num_channel_s // d_ratio),
            nn.ReLU(),
            nn.Conv3d(in_channels=num_channel_s // d_ratio, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(1)  # TODO
        )

    def forward(self, x, debug=False):
        _, _, _, H, W = x.size()

        new_H = H // 2
        new_W = W // 2
        part1 = x[:, :, :, :new_H, :new_W]
        part2 = x[:, :, :, :new_H, new_W:]
        part3 = x[:, :, :, new_H:, :new_W]
        part4 = x[:, :, :, new_H:, new_W:]
        g1 = torch.cat((self.PLE(part1, part2),
                        self.PLE(part1, part3),
                        self.PLE(part1, part4)), 1)
        g2 = torch.cat((self.PLE(part2, part1),
                        self.PLE(part2, part3),
                        self.PLE(part2, part4)), 1)
        g3 = torch.cat((self.PLE(part3, part1),
                        self.PLE(part3, part2),
                        self.PLE(part3, part4)), 1)
        g4 = torch.cat((self.PLE(part4, part1),
                        self.PLE(part4, part2),
                        self.PLE(part4, part3)), 1)
        W_g1 = self.W_spatial(g1)
        part1 = torch.sigmoid(W_g1.expand_as(part1)) * part1
        W_g2 = self.W_spatial(g2)
        part2 = torch.sigmoid(W_g2.expand_as(part2)) * part2
        W_g3 = self.W_spatial(g3)
        part3 = torch.sigmoid(W_g3.expand_as(part3)) * part3
        W_g4 = self.W_spatial(g4)
        part4 = torch.sigmoid(W_g4.expand_as(part4)) * part4
        part13 = torch.cat((part1, part3), dim=-2)
        part24 = torch.cat((part2, part4), dim=-2)
        merged_tensor = torch.cat((part13, part24), dim=-1)

        return merged_tensor


if __name__ == '__main__':
    import torch

    s_ratio = 8
    c_ratio = 8
    d_ratio = 8
    height = 128
    width = 128
    model = PLE_Module(512, (height // 16) * (width // 16), s_ratio=s_ratio, d_ratio=d_ratio)
    input = torch.randn(8, 512, 8, 8, 8)
    output = model(input)
    print(output.shape)
