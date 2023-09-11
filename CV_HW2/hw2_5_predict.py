import torch
import sys
import numpy as np
import os

import cv2
from model import ResModel

from torch.utils.data import Dataset, DataLoader
from random import choice

import torchvision.transforms as T
from PIL import Image


import matplotlib.pyplot as plt

path = os.listdir('test1')
# print(choice(path))
file = os.getcwd() + '\\test1\\' + choice(path)

def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomCrop(204),
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])
    
def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])

class CatDogDataset(Dataset):
    
    def __init__(self, imgs, class_to_int, mode = "train", transforms = None):
        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        
    def __getitem__(self, idx):
        
        image_name = self.imgs
        
        ### Reading, converting and normalizing image
        #img = cv2.imread(DIR_TRAIN + image_name, cv2.IMREAD_COLOR)
        #img = cv2.resize(img, (224,224))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        #img /= 255.
        img = Image.open(file)
        img = img.resize((224, 224))
        
        if self.mode == "train" or self.mode == "val":
        
            ### Preparing class label
            label = self.class_to_int[image_name.split(".")[0]]
            label = torch.tensor(label, dtype = torch.float32)

            ### Apply Transforms on image
            img = self.transforms(img)

            return img, label
        
        elif self.mode == "test":
            
            ### Apply Transforms on image
            img = self.transforms(img)
            return img
            
        
    def __len__(self):
        return len(self.imgs)


def accuracy(preds, trues):
    
    ### Converting preds to 0 or 1
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    
    ### Calculating accuracy by comparing predictions with true labels
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    
    ### Summing over all correct predictions
    acc = np.sum(acc) / len(preds)
    
    return (acc * 100)


def train_one_epoch(train_data_loader):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for images, labels in tqdm(train_data_loader):
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Reseting Gradients
        optimizer.zero_grad()
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
        
        #Backward
        _loss.backward()
        optimizer.step()
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    train_logs["loss"].append(epoch_loss)
    train_logs["accuracy"].append(epoch_acc)
    train_logs["time"].append(total_time)
        
    return epoch_loss, epoch_acc, total_time

def val_one_epoch(val_data_loader, best_val_acc):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for images, labels in tqdm(val_data_loader):
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    val_logs["loss"].append(epoch_loss)
    val_logs["accuracy"].append(epoch_acc)
    val_logs["time"].append(total_time)
    
    ###Saving best model
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(),"resnet50_best.pth")
        
    return epoch_loss, epoch_acc, total_time, best_val_acc



if __name__ == '__main__':

    class_to_int = {"dog" : 0, "cat" : 1}
    int_to_class = {0 : "dog", 1 : "cat"}


    net = ResModel(2)
    net.load_state_dict(torch.load('resnet50_final2.pth'))
    cls_list = ['cats', 'dogs']

    test_dataset = CatDogDataset(file, class_to_int, mode = "test", transforms = get_val_transform())
    
    test_data_loader = DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = True
    )

    # print(test_dataset)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    net.to(device)

    net.eval()

    outlabel = ''

    with torch.no_grad():
        for i, data in enumerate(test_data_loader):
            data = data.to(device)
            output = net(data)
            if output.item() > 0.5:
                outlabel = 'cat'
            else:
                outlabel = 'dog'
            break

    loss_acc_image_dir = os.getcwd() + '\\loss_acc_image'
    loss_acc_image_file = os.listdir(loss_acc_image_dir)
    for i, loss_acc_image in enumerate(loss_acc_image_file):
        plt.subplot(2, 10, i+1)
        plt.imshow(cv2.imread(os.getcwd() + '\\loss_acc_image\\' + loss_acc_image))
        plt.axis('off')

    tensorboard_img = cv2.imread(os.getcwd() + '\\tensorboard.jpg')
    cv2.imshow('tensor', tensorboard_img)

    imgg = cv2.imread(file)
    cv2.imshow(outlabel, imgg)
    plt.show()
    cv2.waitKey(0)