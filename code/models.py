import torch
from torch import nn
from resnet3d import resnet50


class Enhance_models(nn.Module):
    def __init__(self, args):
        super(Enhance_models, self).__init__()
        self.Reg = resnet50(args)
        self.Glo = resnet50(args, rga_mode=True)
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(1024, 1)

    def forward(self, golbal, local):
        L_map = self.Reg(local)
        G_map = self.Glo(golbal)
        w1 = torch.sigmoid(L_map * G_map)
        GL_map = (w1 + 1) * G_map
        W2 = torch.sigmoid((GL_map + L_map) * (GL_map + G_map))
        GL = W2 * (GL_map + L_map) + W2 * (GL_map + G_map)
        out_fc = self.fc(GL)
        out = self.dropout(out_fc)
        return out
