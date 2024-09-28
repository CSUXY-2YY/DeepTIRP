import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Function
from torchvision import models
from torchvision import utils
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms
import pandas as pd
from PIL import Image
from efficientnet_pytorch import EfficientNet

# %%
class MultiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = (self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        #5 for 2nd level label
        #7 for 1st level label
        y1 = self.data_frame.iloc[idx,7]    
        y2 = self.data_frame.iloc[idx,5]
        age = self.data_frame.iloc[idx, 1]
        #age = torch.FloatTensor(age)
        gender = self.data_frame.iloc[idx, 2]
        #gender = torch.FloatTensor(gender)
        
        if self.transform:
            image = self.transform(image)

        return image, y1, y2, age, gender, img_name

# %%
normalize = transforms.Normalize(mean=[0.5115, 0.5115, 0.5115],
                                  std=[0.1316, 0.1316, 0.1316])

train_data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),normalize])

val_test_data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),normalize])

# %%
data_folder = 'C&G_labels/'

# %%
image_datasets = {
    'train': 
    MultiDataset(csv_file = data_folder + 'train.csv',transform = train_data_transforms),
    'validation': 
    MultiDataset(csv_file = data_folder + 'val.csv',transform = val_test_data_transforms),  
    'test': 
    MultiDataset(csv_file = data_folder + 'test.csv',transform = val_test_data_transforms)
}

# %%
dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=128,
                                shuffle=True,
                                num_workers=8,pin_memory=True),
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=16,
                                shuffle=False,
                                num_workers=8,pin_memory=True), 
    
    'test':
    torch.utils.data.DataLoader(image_datasets['test'],
                                batch_size=16,
                                shuffle=False,
                                num_workers=8,pin_memory=True) 
}

# %%
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
device = torch.device("cuda:0")
model.to(device);

# %%
#Set up parameters
criterion1 = nn.CrossEntropyLoss()
num_epochs = 100

# %%
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# %%
prddir = 'Efficientnet_B4_weights/GorC_2_label/'
if not os.path.exists(prddir):
    os.makedirs(prddir)

def training_Efficientnet_B4():

    best_acc = 0.0

    #Set up Adam optimizer
    optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay= 1e-4)

    #Training starts
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        adjust_learning_rate(optimizer, epoch)
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels1, _, _, _, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels1.to(device)
                outputs = model(inputs)
                
                #criterion1 is softmax loss
                loss = criterion1(outputs, labels)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            if phase =='validation':
                if epoch_acc>best_acc:
                    torch.save(model, 'Efficientnet_B4_weights/GorC_2_label/best_weight_E_B4')
                    best_acc = epoch_acc
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))

    # %%
    torch.save(model, 'Efficientnet_B4_weights/GorC_2_label/best_weight_E_B4_final')

    # %%
    model = torch.load('Efficientnet_B4_weights/GorC_2_label/best_weight_E_B4_final')
    model.to(device);

    # %%
    model.eval();

    # %%
    softM = torch.nn.Softmax(1)
    labels_result = []
    preds_result = []
    outputs_result = []
    all_image_name = []
    running_corrects = 0
    for inputs, labels, _, _, _, img_name in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        labels_result.append(labels.data.detach().cpu().numpy())
        preds_result.append(preds.data.detach().cpu().numpy())
        outputs_result.append(softM(outputs).data.detach().cpu().numpy())
        all_image_name.append(img_name)
    epoch_acc = running_corrects.double() / len(image_datasets['test'])
    print(epoch_acc.data.cpu().numpy())
    labels_result = np.hstack(labels_result)
    preds_result = np.hstack(preds_result)
    outputs_result = np.vstack(outputs_result)
    total = []
    for i in all_image_name:
        total += i
    all_image_name = np.array(total)

    # %%
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(labels_result, outputs_result[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    # %%
    labels_result = np.expand_dims(labels_result, axis=1)
    preds_result = np.expand_dims(preds_result, axis=1)
    all_image_name = np.expand_dims(all_image_name, axis=1)
    final_result = np.hstack([all_image_name,labels_result,preds_result,outputs_result])

    # %%
    np.save('results/E_B4_result',final_result)

if __name__ == '__main__':
    
    training_Efficientnet_B4()