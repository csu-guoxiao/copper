import torch.utils.data as data
import torch
import os
import json
import cv2
import numpy as np

class Oredataset(data.Dataset):
    def __init__(self, json_path, TrainValTest, transform=None):
        assert TrainValTest in ['train', 'val', 'test']
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        self.data_path = json_data[TrainValTest]
        self.transform = transform
        self.cls = ['ore', 'waste']

    def __getitem__(self, index):

        img_low = cv2.imread(os.path.join(self.data_path[index], 'low.tiff'), cv2.IMREAD_UNCHANGED)
        img_high = cv2.imread(os.path.join(self.data_path[index], 'high.tiff'), cv2.IMREAD_UNCHANGED)
        
        
        # Data augmentation
        if self.transform == 'train':
            # Random horizontal flip
            if np.random.rand() < 0.5:
                img_low = cv2.flip(img_low, 1)
                img_high = cv2.flip(img_high, 1)

            # Random vertical flip
            if np.random.rand() < 0.5:
                img_low = cv2.flip(img_low, 0)
                img_high = cv2.flip(img_high, 0)

            # Adjust the brightness and contrast
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2) 
                beta = np.random.uniform(-20, 20) 
                img_low = cv2.convertScaleAbs(img_low, alpha=alpha, beta=beta)
                img_high = cv2.convertScaleAbs(img_high, alpha=alpha, beta=beta)


        img_low = cv2.resize(img_low, (40, 40))
        mean_low = img_low.mean()
        std_low = img_low.std()
        img_low = (img_low - mean_low) / (std_low + 1e-5)
        input_low = torch.from_numpy(img_low).float()
        input_low = input_low.unsqueeze(0)

        img_high = cv2.resize(img_high, (40, 40))
        mean_high = img_high.mean()
        std_high = img_high.std()
        img_high = (img_high - mean_high) / (std_high + 1e-5)
        input_high = torch.from_numpy(img_high).float()
        input_high = input_high.unsqueeze(0)
        
        if 'ore' in self.data_path[index].split('/')[-1]:
            label = torch.tensor(1).long()
        else:
            label = torch.tensor(0).long()
        
        return input_low, input_high, label 
        
    def __len__(self):
        return len(self.data_path)
    
class Oredataset_vis(data.Dataset):
    def __init__(self, json_path, TrainValTest, transform=None):
        assert TrainValTest in ['train', 'val']
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        self.data_path = json_data[TrainValTest]
        self.transform = transform
        self.cls = ['ore', 'waste']

    def __getitem__(self, index):

        img_low = cv2.imread(os.path.join(self.data_path[index], 'low.tiff'), cv2.IMREAD_UNCHANGED)
        img_high = cv2.imread(os.path.join(self.data_path[index], 'high.tiff'), cv2.IMREAD_UNCHANGED)
        
        if self.transform is not None:
            img_low = self.transform(img_low)
            img_high = self.transform(img_high)

        img_low = cv2.resize(img_low, (40, 40))
        mean_low = img_low.mean()
        std_low = img_low.std()
        img_low = (img_low - mean_low) / (std_low + 1e-5)
        input_low = torch.from_numpy(img_low).float()
        input_low = input_low.unsqueeze(0)

        img_high = cv2.resize(img_high, (40, 40))
        mean_high = img_high.mean()
        std_high = img_high.std()
        img_high = (img_high - mean_high) / (std_high + 1e-5)
        input_high = torch.from_numpy(img_high).float()
        input_high = input_high.unsqueeze(0)
        
        if 'ore' in self.data_path[index].split('/')[-1]:
            label = torch.tensor(1).long()
        else:
            label = torch.tensor(0).long()
        
        return input_low, input_high, label, self.data_path[index]
        
    def __len__(self):
        return len(self.data_path)