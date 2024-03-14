# Time-series Medical Image Classification
Image Classification Model for Temporal Disease Progression of Chest X-ray dataset


## Introduction
Disease progression modeling (DPM) uses mathematical functions and scientific principles to describe the quantitative progression of a disease over time, providing valuable insights for the development and use of medicines. Disease Progression models have the potential to improve patient outcomes, reduce healthcare costs, and accelerate the development of new treatments. One such task is to predict the three states of disease progression (improving, stable, or worsening) given the current and past multi-image frontal chest X-ray images. This work focuses on fine-tuning and evaluating the pre-trained Torch X-ray Vision model Cohen et al. (2021) for the temporal image classification task.

## Setup
1. Download the Chest X-ray dataset here: https://drive.google.com/file/d/1rKGrW57Nr6AN-jOQOPOQzQiHk8LD3q2o/view?usp=sharing
2. Download the [MS-CXR-T dataset](https://physionet.org/content/ms-cxr-t/1.0.0/) and move `MS_CXR_T_temporal_image_classification_v1.0.0.csv` to the root directory
3. Run `python feature_extraction.py {feature_extractor}` to extract features from any of the four extractors used
4. Run `python logistic_regression_baseline.py {feature_extractor}` to run the baseline logistic regression model
5. Run `python model2.py {feature_extractor}` to run the second model, five separate independent classifiers
6. Run `python model3.py {feature_extractor}` to run the third model, a combined model with loss masking
