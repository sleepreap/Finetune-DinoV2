import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DinoV2.model import Dinov2Finetuner
from DinoV2.dataset import SegmentationDataModule
import DinoV2.config as config
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from colorPalette import color_palette
from colorPalette import apply_palette

def dataset_predictions(dataloader):
    pred_set = []
    label_set=[]
    prog_bar = tqdm(dataloader, desc="Doing predictions", total=len(dataloader))
    for data in prog_bar:
        original_images = data["original_images"]
        original_lables=data['original_segmentation_maps']
        outputs = model(
        pixel_values=data["pixel_values"].to(device),
        labels= data["labels"].to(device),
        )
        downsampled_logits = torch.nn.functional.interpolate(outputs.logits,
                                                   size=[640,640],
                                                   mode="bilinear", align_corners=False)
        downsampled_map = downsampled_logits.argmax(dim=1)
        for i in range(downsampled_map.size(0)):  # Iterate over each image in the batch
            pred_map = downsampled_map[i].squeeze().detach().cpu().numpy()
            pred_set.append(pred_map)  # Append each image individually to pred_set
        for label in original_lables:
            label_set.append(label)
    return pred_set, label_set

def savePredictions(pred_set, label_set, save_path):
    palette = color_palette()
    for i in tqdm(range(len(pred_set)), desc="Saving predictions"):
        file_name = f"result_{i}"
        pred = pred_set[i]
        label = label_set[i]
        colored_image = apply_palette(pred, palette)
        colored_label = apply_palette(label, palette)
        f, axs = plt.subplots(1, 2)
        f.set_figheight(30)
        f.set_figwidth(50)
        axs[0].set_title("Prediction", {'fontsize': 40})
        axs[0].imshow(colored_image)
        axs[1].set_title("Ground truth", {'fontsize': 40})
        axs[1].imshow(colored_label)
        # Save the figure
        file_path = os.path.join(save_path, f"{file_name}.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(f)

    print("Predictions saved")


if __name__=="__main__":
    data_module = SegmentationDataModule(dataset_dir=config.DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--model_path',
    type=str,
    default='',
    help="Enter the path of your model.ckpt file"
    )
    parser.add_argument(
    '--save_path',
    type=str,
    default='',
    help="enter the path to save your images"
    )

    args = parser.parse_args()
    model_path = os.path.join( '..', args.model_path)
    save_path = os.path.join( '..', args.save_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    data_module = SegmentationDataModule(dataset_dir=config.DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    model = Dinov2Finetuner.load_from_checkpoint(model_path,id2label=config.ID2LABEL, lr=config.LEARNING_RATE)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
    pred_set, label_set= dataset_predictions(test_dataloader)
    savePredictions(pred_set, label_set, save_path)
    
