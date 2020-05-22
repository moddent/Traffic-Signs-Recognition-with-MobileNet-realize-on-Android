from torch.utils.data import DataLoader
from torchvision import transforms
from My_data_set import MyDataset


# Hyper parameter
TRAIN_BATCH_SIZE = 10
TEST_BATCH_SIZE = 10
TRAIN_FOLDER = "TrainingSet"
TEST_FOLDER = "TestingSet"


# prepare data
hand_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = MyDataset(transform=hand_transforms, folder_path=TRAIN_FOLDER)
train_data_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
print("n of Training Set:", len(train_data_loader.dataset))

test_data = MyDataset(transform=hand_transforms, folder_path=TEST_FOLDER)
test_data_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=True)
print("n of Testing Set:", len(test_data_loader.dataset))


for step, (x, y) in enumerate(train_data_loader):
    print(len(train_data_loader.dataset))
    print(x.size())
    print(y)
    break


for step, (x, y) in enumerate(test_data_loader):
    print(len(test_data_loader.dataset))
    print(x.size())
    print(y)
    break
