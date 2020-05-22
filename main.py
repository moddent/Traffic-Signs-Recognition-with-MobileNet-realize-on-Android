import time

import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from MobileNet import MobileNet
from My_data_set import MyDataset

# Hyper parameter
TRAIN_BATCH_SIZE = 10
TEST_BATCH_SIZE = 10
TRAIN_FOLDER = "TrainingSet"
TEST_FOLDER = "TestingSet"
EPOCH = 100
LR = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
# prepare data
traffic_sign_transforms = transforms.Compose([transforms.Resize([256, 256]),
					      transforms.RandomAffine(degrees=(-30, 30),
                                                                      translate=(0, 0.5),
                                                                      scale=(0.3, 1),
                                                                      shear=(-30, 30),
                                                                      fillcolor=(255, 255, 255),
                                                                      resample=PIL.Image.BILINEAR,),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

train_data = MyDataset(transform=traffic_sign_transforms, folder_path=TRAIN_FOLDER)
train_data_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# print("n of Training Set:", len(train_data_loader.dataset))

test_data = MyDataset(transform=traffic_sign_transforms, folder_path=TEST_FOLDER)
test_data_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=True)
# print("n of Testing Set:", len(test_data_loader.dataset))


print("# Device:", device, "|Training Size:", len(train_data_loader.dataset)
      , "|Testing Size:", len(test_data_loader.dataset))

# load model
try:
    mobilenet = torch.load('mobile.pt')
    print("# Load model complete")
except:
    mobilenet = MobileNet(class_num=43).to(device)
    print("# start with new model")
time.sleep(1)

optimizer = torch.optim.Adam(mobilenet.parameters(), lr=LR)
# optimizer = torch.optim.SGD(mobilenet.parameters(), lr=LR, momentum=0.9, nesterov=True)
scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
loss_func = nn.CrossEntropyLoss()

best_accuracy = .0

for epoch in range(1, 1 + EPOCH):

    running_results = {'batch_sizes': 0, 'loss': 0}
    train_bar = tqdm(train_data_loader)
    for b_x, b_y in train_bar:
        batch_size = b_x.size(0)
        running_results['batch_sizes'] += batch_size

        mobilenet.train()
        b_x = b_x.to(device)
        b_y = b_y.to(device)

        output = mobilenet(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss for current batch before optimization
        running_results['loss'] += loss.item() * batch_size

        correct = 0
        accuracy = 0
        save_flag = False

        if running_results['batch_sizes'] == len(train_data_loader.dataset):
            mobilenet.eval()
            for _, (t_x, t_y) in enumerate(test_data_loader):
                t_x = t_x.to(device)
                t_y = t_y.cpu().detach().numpy()
                test_output = mobilenet(t_x)
                pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
                for i in range(len(t_x)):
                    if pred_y[i] == t_y[i]:
                        correct += 1

            accuracy = correct / len(test_data_loader.dataset)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(mobilenet, "mobile.pt")
                save_flag = True

        train_bar.set_description(desc='[%d/%d] | Loss:%.4f | Accuracy:%.4f | Save Model:%r | ' %
                                  (epoch, EPOCH, running_results['loss'] / running_results['batch_sizes'],
                                   accuracy, save_flag))
    scheduler.step(epoch=None)
