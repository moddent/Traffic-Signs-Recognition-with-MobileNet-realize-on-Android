import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path)


class MyDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=default_loader, folder_path=None):

        '''
            ClassId	SignName
        0	Speed limit (20km/h)
        1	Speed limit (30km/h)
        2	Speed limit (50km/h)
        3	Speed limit (60km/h)
        4	Speed limit (70km/h)
        5	Speed limit (80km/h)
        6	End of speed limit (80km/h)
        7	Speed limit (100km/h)
        8	Speed limit (120km/h)
        9	No passing
        10	No passing for vehicles over 3.5 metric tons
        11	Right-of-way at the next intersection
        12	Priority road
        13	Yield
        14	Stop
        15	No vehicles
        16	Vehicles over 3.5 metric tons prohibited
        17	No entry
        18	General caution
        19	Dangerous curve to the left
        20	Dangerous curve to the right
        21	Double curve
        22	Bumpy road
        23	Slippery road
        24	Road narrows on the right
        25	Road work
        26	Traffic signals
        27	Pedestrians
        28	Children crossing
        29	Bicycles crossing
        30	Beware of ice/snow
        31	Wild animals crossing
        32	End of all speed and passing limits
        33	Turn right ahead
        34	Turn left ahead
        35	Ahead only
        36	Go straight or right
        37	Go straight or left
        38	Keep right
        39	Keep left
        40	Roundabout mandatory
        41	End of no passing
        42	End of no passing by vehicles over 3.5 metric tons
        '''

        file_path = folder_path
        print("# Preparing imgs in "+folder_path)
        imgs = []
        for file in os.listdir(file_path):
            imgs.append((file_path + "/" + file, int(file[:2])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # print(fn, label)
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
