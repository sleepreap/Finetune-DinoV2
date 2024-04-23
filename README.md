# Finetuning DinoV2 on custom Dataset

## Introduction
DINOV2: A Self-supervised Vision Transformer Model

A family of foundation models producing universal features suitable for image-level visual tasks (image classification, instance retrieval, video understanding) as well as pixel-level visual tasks (depth estimation, semantic segmentation).

### [DINOV2 Project page](https://github.com/facebookresearch/dinov2) | [DINOV2 Paper] (https://arxiv.org/abs/2304.07193) | 
### [Run our demo: ](https://dinov2.metademolab.com/)
### [DINOV2 Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/dinov2)

## Purpose
The purpose of this document is to build a process of finetuning DINOv2 for custom dataset on semantic segmentation. The code is done using Pytorch Lightning and the model can be imported from hugging face.

1. Create a virtual environment: `conda create -n DinoV2 python=3.10 -y` and `conda activate DinoV2 `
2. Install [Pytorch CUDA 12.1](https://pytorch.org/): ` pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 `
3. Download code: `git clone https://github.com/sleepreap/Finetune-DinoV2.git` 
4. `cd Finetune-DinoV2` and run `pip install -e .`

## Dataset
Use createDataset.py to create the folders.
Refer to the README file in the folder "Data" on where to upload the images and labels.

## Training
1. 'cd scripts' 
2. set up the configs required in config.py
3. run the train.py file

A CSVlogger and the trained model file will be saved after the training has been completed. The model file would be saved as "DinoV2.ckpt" in the same directory. An "output" folder will be created to store the contents of the CSVlogger.

## Testing
The testing is done using Mean-IOU, as well as pixel accuracy from the evaluate package. It will provide individual accuracy and IOU scores for each class label specified, as well as the mean accuracy and IOU scores of all the class labels. To run the test file, the model path of the trained model must be provided as an argument.

1. 'cd scripts' 
2. run the test.py file using this command: python test.py --model_path MODEL_PATH
   
```bash
e.g python test.py --model_path DinoV2.ckpt
```

## Utilities
This folder contains the following scripts:
1. inference.py
2. saveComparison.py
3. predictionOverlay.py
4. saveComparisonWithOverlay.py
   
### All the scripts already reference the parent folder for the command line arguments. The images and labels used are from the test dataset respectively.

Inference.py would save all the predictions by the model on the test dataset in the specified save path folder.



```bash
1. 'cd scripts/utilities'
2. run the inference.py file using this command: python inference.py --model_path MODEL_PATH --save_path SAVE_PATH
```

saveComparison.py would save a plot of the prediction and the ground truth side by side in the specified save path folder. There is only 1 comparison per image due to memory constraint.

```bash
1. 'cd scripts/utilities'
2. run the saveComparison.py file using this command: python saveComparison.py --model_path MODEL_PATH --save_path SAVE_PATH
```
predictionOverlay.py would save the overlay that shows the TP+TN+FP+FN of the predictions done by the model for all the images in the specified save path folder. Black means TN (background), Green means TP (metal-line), Red means FN (metal-line as background), Blue means FP (background as metal-line).

```bash
1. 'cd scripts/utilities'
2. run the predictionOverlay.py file using this command: python predictionOverlay.py --model_path MODEL_PATH --save_path SAVE_PATH
```
saveComparisonWithOverlay.py would save a plot of the overlay and the ground truth side by side in the specified save path folder. There is only 1 comparison per image due to memory constraint.

```bash
1. 'cd scripts/utilities'
2. run the saveComparisonWithOverlay.py file using this command: python saveComparisonWithOverlay.py --model_path MODEL_PATH --save_path SAVE_PATH
```

## Citation
```BibTeX
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
