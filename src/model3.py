# -*- coding: utf-8 -*-
import skimage
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms

import torchxrayvision as xrv
from tqdm.notebook import tqdm
from glob import glob
from time import time
import numpy as np
import json

# from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from tqdm.notebook import tqdm

import pandas as pd

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

torch.manual_seed(10)

annotations_file = glob('*image*.csv')[0]
BATCH_SIZE = 8

FEATURE_EXTRACTORS = ["densenet121-res224-chex", "densenet121-res224-rsna", "densenet121-res224-mimic_ch",]
feature_extractor = sys.argv[1]
print(feature_extractor)
assert feature_extractor in FEATURE_EXTRACTORS, f"Only {','.join(FEATURE_EXTRACTORS)} are valid feature extractors."


class CXRT_Dataset_2(Dataset):
  DISEASE_NAMES = ['edema', 'consolidation', 'pleural_effusion', 'pneumothorax', 'pneumonia']
  def __init__(self, annotations_file, features_file):
        self.df = pd.read_csv(annotations_file)
        self.image1 = [p.split('/')[-1] for p in self.df.previous_dicom_id.to_numpy()]
        self.image2 = [p.split('/')[-1] for p in self.df.dicom_id.to_numpy()]

        for disease_name in self.DISEASE_NAMES:
          self.df[f'{disease_name}_progression'].fillna(-1, inplace=True)
          self.df[f'{disease_name}_progression'].replace({'worsening': 1, 'stable': 0, 'improving': 2}, inplace=True)

        with open(features_file) as f:
          data = json.load(f)
          self.image_names = [d.split('/')[-1] for d in data.keys()]
          self.in_features = torch.Tensor(list(data.values()))

  def __len__(self):
        return self.df.shape[0]

  def __getitem__(self, idx):
        label = torch.LongTensor([self.df[f'{d}_progression'].iloc[idx] for d in self.DISEASE_NAMES])

        image1 = self.in_features[self.image_names.index(self.image1[idx]+'.jpg')]
        image2 = self.in_features[self.image_names.index(self.image2[idx]+'.jpg')]
        return image1, image2, label

class Net2(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.lin_cat = nn.Linear(in_features=2*in_features, out_features=hidden_features)
        self.lin1 = nn.Linear(in_features=hidden_features, out_features=num_classes)
        self.lin2 = nn.Linear(in_features=hidden_features, out_features=num_classes)
        self.lin3 = nn.Linear(in_features=hidden_features, out_features=num_classes)
        self.lin4 = nn.Linear(in_features=hidden_features, out_features=num_classes)
        self.lin5 = nn.Linear(in_features=hidden_features, out_features=num_classes)

    def forward(self, x1, x2):
        x = torch.cat( (x1, x2), dim=1)
        x = F.relu(self.lin_cat(x))
        x = torch.cat( (self.lin1(x), self.lin2(x),self.lin3(x), self.lin4(x),
                          self.lin5(x)), )
        return x


class Model2(Net2):
    BATCH_SIZE = 8
    def __init__(self, annotations_file, feature_extractor, masking=True, hidden_features=2048):

      dataset = CXRT_Dataset_2(annotations_file, f'features_{feature_extractor}.json')
      self.in_features = dataset.in_features.shape[-1] 
      super().__init__(self.in_features, hidden_features, num_classes=3)

      trainset, testset = torch.utils.data.random_split(dataset, [0.8, 0.2])
      self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=2)
      self.testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
      self.criterion = nn.CrossEntropyLoss()
      self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
      self.masking = masking

    def train(self, num_epochs=50):
      for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
          # running_loss = 0.0
          for i, data in enumerate(self.trainloader, 0):
              image1, image2, labels = data
              labels = labels.flatten()
              mask = labels >= 0 
              labels[~mask] = 0

              # zero the parameter gradients
              self.optimizer.zero_grad()

              logits = self.forward(image1, image2)

              if self.masking:
                log_probs = F.log_softmax(logits)
                losses_flat = -torch.gather(log_probs, dim=1, index=labels.unsqueeze(0)).squeeze()
                losses_flat = losses_flat * mask.float()
                loss = losses_flat.sum() / mask.float().sum()
              else:
                loss = self.criterion(logits, labels)

              loss.backward()
              self.optimizer.step()

    def test(self):
      num_correct, total = 0, 0
      all_preds, all_labels = [], []
      for i, data in tqdm(enumerate(self.testloader, 0)):
          image1, image2, labels = data
          labels = labels.detach().flatten()
          mask = labels >= 0
          labels[~mask] = 0

          outputs = self.forward(image1, image2)
          log_probs = F.log_softmax(outputs)
          preds = np.argmax(outputs.detach().numpy(), axis=1)

          num_correct += (mask*(preds==labels.numpy())).sum().item()
          total += mask.sum().item()
          all_preds += preds.tolist()
          all_labels += labels.tolist()
      print(num_correct, total, 100*num_correct/total)
      print(classification_report(all_preds, all_labels))

    def test_inv(self):
      all_preds, all_labels, all_preds_inv = np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
      for i, data in tqdm(enumerate(self.testloader, 0)):
          image1, image2, labels = data
          labels = labels.detach().flatten()
          mask = labels >= 0

          outputs = net(image1, image2)
          preds = np.argmax(outputs.detach().numpy(), axis=1)

          outputs = net(image2, image1)
          preds_inv = np.argmax(outputs.detach().numpy(), axis=1)

          preds[~mask] = -1
          preds_inv[~mask] = -1

          all_preds = np.concatenate([all_preds, preds])
          all_preds_inv = np.concatenate([all_preds_inv, preds_inv])

      print(confusion_matrix(all_preds, all_preds_inv))

if __name__ == '__main__':
	net = Model2(annotations_file, feature_extractor, 2048)
	for ep in range(10):
		print(f"Epoch {ep*10}")
		net.train(10)
		net.test()
		net.test_inv()
