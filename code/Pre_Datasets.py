import os
import random
import numpy as np
import cv2
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FaceRecognitionDataset(Dataset):
    def __init__(self, root_dir,mode,args,transform=None):
        self.mode = mode
        self.root_dir = root_dir
        # self.root_l = root_l
        # self.val_rate = val_rate
        self.resize = args.resize
        self.transform = transform
        self.train_data = self._split_data()

    def _split_data(self):
        train_images_path, train_images_label = read_split_data(self.root_dir,self.mode)
        train_data = list(zip(train_images_path,train_images_label))
        return train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        image_path,label = self.train_data[index]
        label=int(label)
        frames,frames_l = self._load_frames(image_path)
        if self.mode == 'train':
            frames = self.randomflip(frames)
            frames_l = self.randomflip(frames_l)
        frames = self.normalize(frames)
        frames = self.to_tensor(frames)
        frames_l = self.normalize(frames_l)
        frames_l = self.to_tensor(frames_l)
        return frames,frames_l, label
    def img_pad(self,pil_file):
        w, h, c = pil_file.shape
        fixed_size = 256  # 输出正方形图片的尺寸

        if h <= w:
            factor = w / float(fixed_size)
            new_h = int(h / factor)
            if new_h % 2 != 0:
                new_h -= 1
            pil_file = cv2.resize(pil_file, (new_h, fixed_size))
            pad_h = int((fixed_size - new_h) / 2)
            array_file = np.array(pil_file)
            # array_file = np.pad(array_file, ((0, 0), (pad_w, fixed_size-pad_w)), 'constant')
            array_file = cv2.copyMakeBorder(array_file, 0, 0, pad_h, fixed_size - new_h - pad_h, cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))
        else:
            factor = h / float(fixed_size)
            new_w = int(w / factor)
            if new_w % 2 != 0:
                new_w -= 1
            pil_file = cv2.resize(pil_file, (fixed_size, new_w))
            pad_w = int((fixed_size - new_w) / 2)
            array_file = np.array(pil_file)
            # array_file = np.pad(array_file, ((pad_h, fixed_size-pad_h), (0, 0)), 'constant')
            array_file = cv2.copyMakeBorder(array_file, pad_w, fixed_size - new_w - pad_w, 0, 0, cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))
        output_file = Image.fromarray(array_file)
        return output_file
    def _load_frames(self, video_path):
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        buffer = np.empty((frame_count, self.resize, self.resize, 3), np.dtype('float32'))
        buffer_l = np.empty((frame_count, self.resize, self.resize, 3), np.dtype('float32'))
        count = 0
        retaining = True
        while count < frame_count and retaining:
            retaining, frame = capture.read()
            if retaining is False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            crop_local = frame[53: 181, 55: 215]
            b, g, r = cv2.split(crop_local)  # 通道分离,再重新合并操作
            pil_image = cv2.merge([r, g, b])
            PL_frame = self.img_pad(pil_image)
            PL_frame = np.array(PL_frame)
            frame = cv2.resize(frame, (self.resize, self.resize))
            PL_frame = cv2.resize(PL_frame, (self.resize, self.resize))
            buffer[count] = frame
            buffer_l[count] = PL_frame
            count += 1
        capture.release()
        return buffer[:16],buffer_l[:16]



    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        augmented_buffer = buffer.copy()
        for i in range(buffer.shape[0]):
            if np.random.random() < 0.25:
                augmented_buffer[i] = cv2.flip(augmented_buffer[i], flipCode=1)
            if np.random.random() < 0.25:
                angle = random.uniform(-10, 10)
                rows, cols, _ = augmented_buffer[i].shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                augmented_buffer[i] = cv2.warpAffine(augmented_buffer[i], M, (cols, rows))
            if np.random.random() < 0.25:
                augmented_buffer[i] = cv2.flip(augmented_buffer[i], flipCode=0)
        return augmented_buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]])) / 128.0
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))


def read_split_data(base_root,mode):
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(base_root), "dataset root: {} does not exist.".format(base_root)
    # 遍历文件夹，一个文件夹对应一个类别
    if mode =='train':
        mode_lists = ['dev', 'train']
    else:
        mode_lists = ['test']
    for mode_list in mode_lists:
        root=os.path.join(base_root, mode_list)
        labels = [label for label in os.listdir(root) if os.path.isdir(os.path.join(root, label))]
        # 遍历每个文件夹下的文件
        for label in labels:
            label_path = os.path.join(root, label)
            vids = [os.path.join(root, label, vid) for vid in os.listdir(label_path)]
            for vid in vids:
                vid_path=os.path.join(label_path, vid)
                for seq in os.listdir(vid_path):
                    seq_path=os.path.join(label_path, vid, seq)
                    capture = cv2.VideoCapture(seq_path)
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count>15:
                        train_images_path.append(seq_path)
                        train_images_label.append(label)

    print("{} images were found in the dataset.".format(len(train_images_path)))
    print("{} images for training.".format(len(train_images_path)))
    assert len(train_images_path) > 0, "number of training images must be greater than 0."
    return train_images_path, train_images_label





