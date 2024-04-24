import pytorch_lightning as pl
import torch
from pathlib import Path
torch.set_float32_matmul_precision("medium")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DinoV2 import ( Dinov2Finetuner, 
                        SegmentationDataModule, 
                        DATASET_DIR, 
                        BATCH_SIZE, 
                        NUM_WORKERS, 
                        ID2LABEL, 
                        LEARNING_RATE)
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from colorPalette import color_palette, apply_palette

def dataset_predictions(dataloader):
    pred_set = []
    prog_bar = tqdm(dataloader, desc="Doing predictions", total=len(dataloader))
    for data in prog_bar:
      original_images = data["original_images"]
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
      #pred_maps=downsampled_map.squeeze().detach().cpu().numpy()
      #pred_set.append(pred_maps)
    return pred_set
    


def savePredictions(pred_set, save_path):
    palette = color_palette()
    for i in tqdm(range(len(pred_set)), desc="Saving predictions"):
        file_name = f"result_{i}"
        colored_image = apply_palette(pred_set[i], palette)
        plt.imshow(colored_image)  
        plt.axis('off')  # Turn off axis numbers and ticks
        # Construct the full path where the image will be saved
        file_path = os.path.join(save_path, f"{file_name}.png")
        # Save the figure
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
    print("Predictions saved")


if __name__=="__main__":
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
        
    data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = Dinov2Finetuner.load_from_checkpoint(model_path,id2label=ID2LABEL, lr=LEARNING_RATE)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
    pred_set= dataset_predictions(test_dataloader)
    savePredictions(pred_set, save_path)
        
