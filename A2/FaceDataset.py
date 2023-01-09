import os
import cv2
import dlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Config
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from imutils import face_utils


class FaceDataset(Dataset):
    def __init__(self, cfg, split):
        self.data_dir = cfg.dataset_dir
        if (split == 'test'):
            self.data_dir += '_test'
        self.split = split
        self.transform = transforms.ToTensor()
        self.img_list, self.label_list = self._load_image_info()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_file = os.path.join(self.data_dir, 'img', img_name)
        image = self._read_image(img_file)
        label = (self.label_list[idx] == 1)

        density = self.detect_face(image)
        density = np.expand_dims(density, axis=-1)
        inmap = np.concatenate((image, density), axis=2)

        inmap = self.transform(inmap)
        sample = {'image': inmap, 'label': label}
        return sample

    def _load_image_info(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'labels.csv'), sep='\t')
        img_list = df['img_name'].values.tolist()
        label_list = df['smiling'].values.tolist()
        return img_list, label_list

    def _read_image(self, image_file):
        image = cv2.imread(image_file)
        return image

    def gaussian_filter_density(self, gt, pts):
        density = np.zeros(gt.shape, dtype=np.float32)
        for i, pt in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            sigma = 2
            density += gaussian_filter(pt2d, sigma, mode='constant')
        density = density * 255.
        return density

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray, 1)

        if (len(faces) == 0):
            density = np.ones(gray.shape, dtype=np.float32)
        else:
            shape = self.predictor(image, faces[0])
            points = face_utils.shape_to_np(shape)
            mouse_points = points[48:68]
            density = self.gaussian_filter_density(gray, mouse_points)
        return density


if __name__ == '__main__':
    cfg = Config
    dataset = FaceDataset(cfg, split='train')
    data_loader = DataLoader(dataset, cfg.batch_size, num_workers=cfg.num_works, shuffle=True)
    for sample_batched in tqdm(data_loader):  # Iterate over multiple batches in the entire dataset
        images, labels = sample_batched['image'], sample_batched['label']
        print(labels)
