# Skin Disease Classification

This project is for learning image classification on skin disease pictures.
It is not a medical tool.
The model can make mistakes, so do not use it for diagnosis.

## What this project does

You upload a skin image and the app predicts the most likely skin condition.
The current model uses 9 skin classes.
I used this project to learn how to improve accuracy from the validation folder in the dataset.

## The classes are

Actinic keratosis
Atopic Dermatitis
Benign keratosis
Dermatofibroma
Melanocytic nevus
Melanoma
Squamous cell carcinoma
Tinea Ringworm Candidiasis
Vascular lesion

## What to expect

The accuracy is not fixed.
It depends on the dataset quality, class balance, training settings, and whether TensorFlow can use the GPU in WSL.
This project is mainly for learning and improving accuracy, not for real medical use.

## Main files

train.py trains the model
test.py checks the model on the validation split
app1.py runs the web app
download_dataset.py downloads the dataset
simple_rebalance.py helps balance the classes

## How the training works

Training uses datasets/train
Validation and testing are taken from datasets/val
There is no separate test folder in the current setup
The model uses MobileNetV3Large
Images are used in raw form, so do not rescale them before training or inference

## Setup

1. Open WSL2 on Windows


2. Go to the project folder

- bash
  - cd /mnt/d/Code/Skin-disease-prediction


3. Create and activate the virtual environment

- bash
  - python3 -m venv venv
  - source venv/bin/activate


4. Install the packages

- bash
  - pip install "tensorflow[and-cuda]"
  - pip install -r requirements.txt


If GPU is not detected

Run this before training

- bash
  - export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/curand/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusolver/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/nccl/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH


## Dataset

Download the dataset with

- bash
  - python download_dataset.py



Training

 - bash
  - python train.py


The model trains in two phases. First it trains the new head. Then it fine tunes the top layers.

## Testing

- bash
  - python test.py


This checks the model using the holdout part of the val folder and prints accuracy, macro F1, weighted F1, and the confusion matrix.

## Web app

- bash
  - python app1.py


Then open

text
http://localhost:5000


The app shows the top prediction, the top 3 predictions, a confidence chart, and a short condition description.

## Example workflow

 - bash
   - python download_dataset.py
   - python simple_rebalance.py
   - python train.py
   - python test.py
   - python app1.py
