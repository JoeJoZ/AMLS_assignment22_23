import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Config
from tqdm import tqdm

class FaceDataset(Dataset):
    def __init__(self, cfg, split):
        self.data_dir = cfg.dataset_dir
        if (split == 'test'):
            self.data_dir += '_test'
        self.split = split
        self.transform = transforms.ToTensor()
        self.img_list, self.label_list = self._load_image_info()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_file = os.path.join(self.data_dir, 'img', img_name)
        image = self._read_image(img_file)
        label = (self.label_list[idx] == 1)

        image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample

    def _load_image_info(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'labels.csv'), sep='\t')
        img_list = df['img_name'].values.tolist()
        label_list = df['gender'].values.tolist()
        return img_list, label_list

    def _read_image(self, image_file):
        image = cv2.imread(image_file)
        return image


if __name__ == '__main__':
    cfg = Config
    dataset = FaceDataset(cfg, split='test')
    data_loader = DataLoader(dataset, cfg.batch_size, num_workers=cfg.num_works, shuffle=True)
    for sample_batched in tqdm(data_loader):  # 遍历整个数据集中多个 batch
        images, labels = sample_batched['image'], sample_batched['label']
        print(labels)
