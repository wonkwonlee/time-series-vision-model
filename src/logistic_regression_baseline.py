import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms

from tqdm.notebook import tqdm
from glob import glob
from time import time
import numpy as np
import json

from tqdm.notebook import tqdm

import pandas as pd
from sklearn.metrics import classification_report, confusion matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split


FEATURE_EXTRACTORS = ["densenet121-res224-chex", "densenet121-res224-rsna", "densenet121-res224-mimic_ch",]
feature_extractor = sys.argv[1]
print(feature_extractor)
assert feature_extractor in FEATURE_EXTRACTORS, f"Only {','.join(FEATURE_EXTRACTORS)} are valid feature extractors."

with open(f'features_{feature_extractor}.json') as f:
  data = json.load(f)
images = list(data.keys())
features = torch.Tensor(list(data.values()))


"""Train the model"""

annotations_file = glob('*image*.csv')[0]

DISEASE_NAMES = ['edema', 'consolidation', 'pleural_effusion', 'pneumothorax', 'pneumonia']
for disease_name in DISEASE_NAMES:
    print("Disease:", disease_name)
    df = pd.read_csv(annotations_file)
    df = df[~df[f'{disease_name}_progression'].isna()]
    image1 = df.previous_dicom_id.to_numpy()
    image2 = df.dicom_id.to_numpy()

    df[f'{disease_name}_progression'].replace({'worsening': 1, 'stable': 0, 'improving': 2}, inplace=True)
    labels = df[f'{disease_name}_progression'].to_numpy()

    with open(f'features_{feature_extractor}.json') as f:
      data = json.load(f)
      image_names = [d.split('/')[-1] for d in data.keys()]
      in_features = np.array(list(data.values()))

    features1 = np.array([in_features[image_names.index(img.split('/')[-1]+'.jpg')] for img in image1])
    features2 = np.array([in_features[image_names.index(img.split('/')[-1]+'.jpg')] for img in image2])
    features = np.concatenate([features1, features2], axis=1)
    X, X_test, y, y_test = train_test_split(features, labels, test_size=0.2, random_state=123, shuffle=True)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X, y)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    ### EXPERIMENT 2:
    print("=========== EXP2 =============================")
    feat1 = X_test[:, :X_test.shape[1]//2+1]
    feat2 = X_test[:, X_test.shape[1]//2+1:]
    inv_feat = np.concatenate([feat2, feat1], axis=1)
    preds_inv = model.predict(inv_feat)
    y_test_inv = y_test.copy()
    y_test_inv[y_test==1] = 2
    y_test_inv[y_test==2] = 1

    print(confusion_matrix(preds, preds_inv))
    # print(classification_report(y_test_inv, preds_inv))

