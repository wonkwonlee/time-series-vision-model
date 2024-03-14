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
import pandas as pd

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from tqdm.notebook import tqdm
import json
import csv
import sys

import pandas as pd

FEATURE_EXTRACTORS = ["densenet121-res224-chex", "densenet121-res224-rsna", "densenet121-res224-mimic_ch", "ViT"]
feature_extractor = sys.argv[1]
print(feature_extractor)
assert feature_extractor in FEATURE_EXTRACTORS, f"Only {','.join(FEATURE_EXTRACTORS)} are valid feature extractors."


def extract_vit():
	images = sorted(glob('ms_cxr_t_images/*'))[:]

	processor = ViTImageProcessor.from_pretrained('nickmuchi/vit-finetuned-chest-xray-pneumonia')
	model = ViTModel.from_pretrained('nickmuchi/vit-finetuned-chest-xray-pneumonia')

	features_vit = []
	for image_path in tqdm(images):
	  image = Image.open(image_path)
	  rgbimg = Image.new("RGB", image.size)
	  rgbimg.paste(image)
	  image = rgbimg

	  inputs = processor(images=image, return_tensors="pt")
	  outputs = model(**inputs)

	  last_hidden_states = outputs.last_hidden_state.unsqueeze(0).to(torch.float16).detach()
	  if len(features_vit) == 0:
	    features_vit = last_hidden_states
	  else:
	    features_vit = torch.cat((features_vit, last_hidden_states), 0)

	print(features_vit.squeeze().shape)

	with open('features_vit.csv', 'w+') as f:
	  writer = csv.writer(f)
	  d = list(zip(images, features_vit.squeeze(0).detach().numpy()))
	  writer.writerows(d)

	d = dict(zip(images, features_vit.squeeze().tolist()))
	with open(f'features_vit.json', 'w+') as f:
		json.dump(fp=f, obj=d, indent=2)


def extract_densenet(model_name):
	"""
	  **X-Ray Vision models**
	Change the variable model_name and run all the cells below to get the features in "features_model_name.json". Download it
	"""

	model = xrv.models.get_model(model_name)
	features = []
	X = []

	transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])
	images = sorted(glob('ms_cxr_t_images/*'))

	for img_path in tqdm(images):
	  img = skimage.io.imread(img_path)
	  img = xrv.datasets.normalize(img, 255)
	  if len(img.shape) < 2:
	      print("error, dimension lower than 2 for image")
	  if len(img.shape) > 2:
	    img = img[:, :, 0]
	  img = img[None, :, :]
	  img = transform(img)
	  img = torch.from_numpy(img).unsqueeze(0)
	  img = xrv.models.fix_resolution(img, 224, model)
	  if len(X) == 0:
	    X = img
	  else:
	    X = torch.cat((X, img), dim=0)
	    print(X.shape)

	D = 200
	with torch.no_grad():
	  features = model.features2(X[:D])
	  for i in tqdm(range(D, len(X), D)):
	    features = torch.cat( (features, model.features2(X[i: (i+D)]) ), dim=0)
	print(features.shape)

	d = dict(zip(images, features.tolist()))
	with open(f'features_{model_name}.json', 'w+') as f:
	  json.dump(fp=f, obj=d, indent=2)


def missing_data_stats():
	df = pd.read_csv(glob('*image*.csv')[0])
	df['label'] = 'blah'
	df.label[~df.edema_progression.isna()] = df.edema_progression[~df.edema_progression.isna()]
	print(df.label.value_counts())
	print(df.edema_progression.value_counts())
	df.label[~df.consolidation_progression.isna()] = df.consolidation_progression[~df.consolidation_progression.isna()]
	print(df.label.value_counts())
	print(df.consolidation_progression.value_counts())

	label_cols = [c for c in df.columns if 'progression' in c]
	for i in range(5):
		for j in range(i+1, 5):
			col1, col2 = label_cols[i], label_cols[j]
			print(df[(~df[col1].isna()) & (~df[col2].isna())].shape, '\t', col1, '\t', col2)


if __name__ == "__main__":
	if feature_extractor == "ViT":
		extract_vit()
	else:
		extract_densenet(feature_extractor)

	missing_data_stats()
