import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision import utils
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms
import pandas as pd
from PIL import Image
from sklearn import metrics
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
        #7 for 1st level label
        y = self.data_frame.iloc[idx,7]    
        age = self.data_frame.iloc[idx, 1]
        gender = self.data_frame.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)

        return image, y, age, gender, img_name

# %%
normalize = transforms.Normalize(mean=[0.5115, 0.5115, 0.5115],
                                  std=[0.1316, 0.1316, 0.1316])

train_data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),normalize])

val_test_data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
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
                                batch_size=64,
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
# import timm
# print(timm.models.create_model('swin_base_patch4_window7_224').default_cfg) 

# %%
# import timm
# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

# import timm
# avail_pretrained_models = timm.list_models("*Efficient*")
# avail_pretrained_models

# avail_pretrained_models = timm.list_models("*swin*")
# avail_pretrained_models

# avail_pretrained_models = timm.list_models(pretrained=True)
# len(avail_pretrained_models)

# pretrained_cfg = timm.create_model('swin_base_patch4_window7_224').default_cfg
# pretrained_cfg['file'] = r'/public/home/hxu6/projects/medical_image/wang_hua/pretrained/swin_base_patch4_window7_224_22kto1k.pth'
# model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, pretrained_cfg=pretrained_cfg)

# %%
import timm
NUM_FINETUNE_CLASSES = 2
file = r'Pretrained_model/swin_base_patch4_window7_224_22kto1k.pth'
# model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True,
                          num_classes=NUM_FINETUNE_CLASSES, 
                          pretrained_cfg_overlay=dict(file=file))

# %%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.memory_summary(device=device))
device = torch.device("cuda:0")
model.to(device)

def training_Swin_transformer():
    prddir = 'Swin_Transformer_weights/GorC_2_label/'
    if not os.path.exists(prddir):
        os.makedirs(prddir)

    # %%
    criterion1 = nn.CrossEntropyLoss()
    num_epochs = 100
    lr = 3e-5
    gamma = 0.7
    seed = 42

    # %%
    #Create the model, change the output layer to 3
    best_acc = 0.0

    #Set up Adam optimizer
    #optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay= 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)
    #Training starts
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels1, _, _, _ in dataloaders[phase]:
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
                    torch.save(model, prddir + 'best_weight_ViT')
                    best_acc = epoch_acc
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))

    # %%
    torch.save(model, 'Swin_Transformer_weights/GorC_2_label/best_weight_SwinTrans_final')

    # %%
    model = torch.load('Swin_Transformer_weights/GorC_2_label/best_weight_SwinTrans_final')
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
    for inputs, labels, _, _, img_name in dataloaders['test']:
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
    fpr, tpr, thresholds = metrics.roc_curve(labels_result, outputs_result[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    # %%
    labels_result = np.expand_dims(labels_result, axis=1)
    preds_result = np.expand_dims(preds_result, axis=1)
    all_image_name = np.expand_dims(all_image_name, axis=1)
    final_result = np.hstack([all_image_name,labels_result,preds_result,outputs_result])

    # %%
    np.save('results/SwinTrans_result',final_result)

    # %%


    # %% [markdown]
    # 

    # %% [markdown]
    # ## testing

    # %%
    #Create the model, change the output layer to 3
    best_acc = 0.0

    #Set up Adam optimizer
    #optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay= 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)
    #Training starts
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels1, labels2, _, _, _ in dataloaders[phase]:
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
                    torch.save(model, 'Transformer_weights/GorC_2_label/best_weight_ViT')
                    best_acc = epoch_acc
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))

    # %%
    torch.save(model, 'trained_weights/best_weight_E_B5_771')

    # %%
    #load model
    #model = DenseNet121(3).cuda()
    model = torch.load('SwinTransformer_weights/GorC_2_label/best_weight_swin_final')
    model.to(device);

    # %%
    softM = torch.nn.Softmax(1)
    labels_result = []
    preds_result = []
    outputs_result = []
    running_corrects = 0
    for inputs, labels, _, _, _ in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        labels_result.append(labels.data.detach().cpu().numpy())
        preds_result.append(preds.data.detach().cpu().numpy())
        outputs_result.append(softM(outputs).data.detach().cpu().numpy())
    epoch_acc = running_corrects.double() / len(image_datasets['test'])
    print(epoch_acc.data.cpu().numpy())
    labels_result = np.hstack(labels_result)
    preds_result = np.hstack(preds_result)
    outputs_result = np.vstack(outputs_result)

if __name__ == '__main__':
    
    training_Swin_transformer()
