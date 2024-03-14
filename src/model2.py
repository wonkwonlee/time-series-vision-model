import skimage
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm.notebook import tqdm
from glob import glob
from time import time
import numpy as np
import json

from tqdm.notebook import tqdm
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix

torch.manual_seed(10)

annotations_file = glob('*image*.csv')[0]
BATCH_SIZE = 8

FEATURE_EXTRACTORS = ["densenet121-res224-chex", "densenet121-res224-rsna", "densenet121-res224-mimic_ch",]
feature_extractor = sys.argv[1]
print(feature_extractor)
assert feature_extractor in FEATURE_EXTRACTORS, f"Only {','.join(FEATURE_EXTRACTORS)} are valid feature extractors."


class CXRT_Dataset_1(Dataset):
  DISEASE_NAMES = ['edema', 'consolidation', 'pleural_effusion', 'pneumothorax', 'pneumonia']
  def __init__(self, annotations_file, features_file, disease_name):
        assert disease_name in self.DISEASE_NAMES, "valid disease names: " + " ".join(self.DISEASE_NAMES)
        self.df = pd.read_csv(annotations_file)
        self.df = self.df[~self.df[f'{disease_name}_progression'].isna()]
        self.image1 = self.df.previous_dicom_id.to_numpy()
        self.image2 = self.df.dicom_id.to_numpy()
        self.image1 = [p.split('/')[-1] for p in self.image1]
        self.image2 = [p.split('/')[-1] for p in self.image2]
        self.df[f'{disease_name}_progression'].replace({'worsening': 1, 'stable': 0, 'improving': 2}, inplace=True)

        self.labels = torch.LongTensor(self.df[f'{disease_name}_progression'].to_list())

        with open(features_file) as f:
          data = json.load(f)
          self.image_names = [d.split('/')[-1] for d in data.keys()]
          self.in_features = torch.Tensor(list(data.values()))

  def __len__(self):
        return self.df.shape[0]

  def __getitem__(self, idx):
        label = self.labels[idx]

        image1 = self.in_features[self.image_names.index(self.image1[idx]+'.jpg')]
        image2 = self.in_features[self.image_names.index(self.image2[idx]+'.jpg')]
        return image1, image2, label


class Net(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.lin2 = nn.Linear(in_features=2*in_features, out_features=hidden_features)
        self.lin3 = nn.Linear(in_features=hidden_features, out_features=num_classes)

    def forward(self, x1, x2):
        x = torch.cat( (x1, x2), dim=1)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x))
        return x


class Model1(Net):
    BATCH_SIZE = 8
    def __init__(self, disease_name, annotations_file, feature_extractor, hidden_features=2048):

      dataset = CXRT_Dataset_1(annotations_file, f'features_{feature_extractor}.json', disease_name)
      self.in_features = dataset.in_features.shape[-1]

      super().__init__(self.in_features, hidden_features, num_classes=3)

      trainset, testset = torch.utils.data.random_split(dataset, [0.8, 0.2])

      self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=2)

      self.testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
      self.criterion = nn.CrossEntropyLoss()
      self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def train(self, num_epochs=50):
      for epoch in tqdm(range(num_epochs)):
          running_loss = 0.0
          for i, data in enumerate(self.trainloader, 0):
              image1, image2, labels = data

              self.optimizer.zero_grad()

              outputs = self.forward(image1, image2)
              loss = self.criterion(outputs, labels)
              loss.backward()
              self.optimizer.step()

              running_loss += loss.item()

    def test(self):
      num_correct, total = 0, 0
      for i, data in tqdm(enumerate(self.testloader, 0)):
          image1, image2, labels = data
          outputs = net(image1, image2)
          preds = np.argmax(outputs.detach().numpy(), axis=1)
          labels = labels.detach().numpy()
          num_correct += (preds==labels).sum()
          total += len(labels)
      print(num_correct, total, 100*num_correct/total)

    def test_inv(self):
      num_correct, total = 0, 0
      all_preds, all_labels, all_preds_inv = np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
      for i, data in tqdm(enumerate(self.testloader, 0)):
          image1, image2, labels = data
          outputs = net(image1, image2)
          preds = np.argmax(outputs.detach().numpy(), axis=1)

          outputs = net(image2, image1)
          preds_inv = np.argmax(outputs.detach().numpy(), axis=1)
          # labels = labels.detach().numpy()

          y_test_inv = labels.detach().numpy().copy()
          y_test_inv[labels==1] = 2
          y_test_inv[labels==2] = 1
          labels = y_test_inv

          all_preds = np.concatenate([all_preds, preds])
          all_preds_inv = np.concatenate([all_preds_inv, preds_inv])
          all_labels = np.concatenate([all_labels, labels])

          num_correct += (preds_inv==preds).sum()
          total += len(labels)
      print(confusion_matrix(all_preds, all_preds_inv))



if __name__ == '__main__':
	for disease in CXRT_Dataset_1.DISEASE_NAMES:
	  net = Model1(disease, annotations_file, feature_extractor, 2048)
	  print(disease.upper())
	  net.train()
	  net.test()
	  net.test_inv()

