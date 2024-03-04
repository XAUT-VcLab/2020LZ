import argparse
from torch.utils.data import DataLoader
import torch
import time
import math
import matplotlib.pyplot as plt
from resnet3d import resnet50
import torch.optim as optim
from Pre_Datasets import FaceRecognitionDataset
import sys
import datetime
from torchvision.transforms import ToTensor
import os
parser = argparse.ArgumentParser()

"""
model setting
"""
parser.add_argument('--Epochs', default=500, type=int, metavar='N')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR', dest='lr')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""
train setting
"""
parser.add_argument('--BATCH_SIZE', default=16, type=int, metavar='N')
parser.add_argument('--milestones', default=[50, 100, 200, 300, 400], type=int, metavar='milestones', )
parser.add_argument('--gamma', default=0.9, type=float, metavar='gamma', )
"""
data setting
"""
parser.add_argument('--resize', default=128, type=int, metavar='N')
parser.add_argument('--clip_len', default=16, type=int, metavar='N')
parser.add_argument('--loads', default=[30, 60, 90, 400], type=str)
args = parser.parse_args()
print(device)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        pass
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal
    def __enter__(self):
        sys.stdout = self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
# 获取当前时间并格式化为字符串，作为日志文件名
    if os.path.exists("./prepth") is False:
        os.makedirs("./prepth")
    if os.path.exists("./Pre_logs") is False:
        os.makedirs("./Pre_logs")
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"./Pre_logs/log_output_{current_time}.log"
image_load = f"img_{current_time}.jpg"
if __name__ == '__main__':

    root_dir = './AVEC2013'

    train_data = FaceRecognitionDataset(root_dir, mode='train', transform=ToTensor(),args=args)
    test_data = FaceRecognitionDataset(root_dir, mode='test', transform=ToTensor(),args=args)
    train_loader = DataLoader(train_data, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=8,
                              drop_last=True)  # 训练集
    test_loader = DataLoader(test_data, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)  # 测试集

    torch.manual_seed(1234)
    model = resnet50(args,rga_mode=True)
    model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-5)
    T_max = 200  # 周期
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max,
                                                           eta_min=1e-7,
                                                           last_epoch=- 1)
    criterion_MSE = torch.nn.MSELoss()
    criterion_MAE = torch.nn.L1Loss()
    best_valid = 8.5
    Final_best_Trainloss = 0
    best_MAE = 1e8
    epoch_data = []
    RMSE_data = []
    MAE_data = []

    with Logger(log_filename):
        print(args)
        print(log_filename, image_load)
    for epoch in range(1, args.Epochs + 1):

        start = time.time()
        epoch_loss = 0  # 累计本轮的损失
        model.train()
        for i_batch, (data,locals, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 梯度初始化
            batch_size = data.size(0)
            output = model(data, )
            output = torch.squeeze(output)
            loss = criterion_MSE(output, target.float())
            loss = torch.sqrt(loss)  # RMSE
            combined_loss = loss
            combined_loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += combined_loss.item() * batch_size  # 累加所有损失
        train_loss = epoch_loss / len(train_loader.dataset)
        model.eval()
        loader = test_loader
        MSE_loss = 0.0
        MAE_loss = 0.0
        with torch.no_grad():
            for i_batch, (data, locals, target) in enumerate(loader):
                data, target = data.to(device),target.to(device)
                batch_size = data.size(0)
                preds = model(data)
                preds = torch.squeeze(preds)
                MSE = criterion_MSE(preds, target)
                MAE = criterion_MAE(preds, target)
                MSE_loss += MSE.item() * batch_size
                MAE_loss += MAE.item() * batch_size
        test_MSE = MSE_loss / (len(test_loader.dataset))
        test_MAE = MAE_loss / (len(test_loader.dataset))
        test_RMSE = math.sqrt(test_MSE)
        end = time.time()
        duration = end - start

        with Logger(log_filename):
            print(
                'Epoch {:3d} | Time {:5.2f} sec |Train Loss {:5.2f} | Test RMSE {:5.2f} | Test MAE {:5.2f}'.
                format(epoch, duration, train_loss, test_RMSE, test_MAE))
        # -------------------更新模型------------------------------
        if test_RMSE < best_valid:
            print(f"*****Saved New model_blocks base RMSE *****")
            checkpoint = {
                'state_dict': model.state_dict(),
            }
            torch.save(checkpoint, './prepth/Best_ACC{}.pth'.format(test_RMSE))
            best_valid = test_RMSE
            best_MAE = test_MAE
            Final_best_Trainloss = train_loss
        else:
            pass
        epoch_data.append(epoch)
        RMSE_data.append(test_RMSE)
        MAE_data.append(test_MAE)
    plt.plot(epoch_data, RMSE_data, linewidth=1, label='RMSE')
    plt.plot(epoch_data, MAE_data, linewidth=1, label='MAE')
    plt.legend()
    plt.savefig(image_load)
    plt.show()
    print("-" * 87)
    with Logger(log_filename):
        print(
            'Best get : Train  {:5.4f} | Test RMSE   {:5.4f}| Test MAE {:5.4f}'.
            format(Final_best_Trainloss, best_valid, best_MAE))
