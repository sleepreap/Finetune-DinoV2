import pytorch_lightning as pl
import torch
from transformers import Dinov2Model
#from transformers import Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from torch import nn
import DinoV2.config as config
from torch.nn.functional import interpolate
from torch.nn import CrossEntropyLoss
import evaluate
import time
import json 
import numpy as np

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=46, tokenH=46, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)
        

class Dinov2Finetuner(pl.LightningModule):
    def __init__(self, id2label, lr):
        super(Dinov2Finetuner, self).__init__()
        self.id2label = id2label
        self.num_classes = len(id2label.keys())
        self.lr=lr
        self.model = Dinov2Model.from_pretrained(
            "facebook/dinov2-base",
            id2label=self.id2label,
            num_labels=self.num_classes
        )
         # Freeze dinov2 parameters
        for name, param in self.model.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False
                
        self.classifier = LinearClassifier(self.model.config.hidden_size, 46, 46, self.model.config.num_labels)
        
        evaluate.load
        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
    # use frozen features
        outputs = self.model(pixel_values,
                                output_hidden_states=output_hidden_states,
                                output_attentions=output_attentions)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:,1:,:]
        
        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)
        
        loss = None
        if labels is not None:
          loss_fct = torch.nn.CrossEntropyLoss()
          loss = loss_fct(logits, labels)
        
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def training_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("loss", loss, sync_dist=True, batch_size=config.BATCH_SIZE)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels= batch["labels"],
        )
        loss = outputs.loss
        self.log("loss", loss, sync_dist=True, batch_size=config.BATCH_SIZE)
        return loss
        
    def test_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        outputs = self(
             pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        ground_truth = batch["original_segmentation_maps"]
        ### Downsample prediction from 644x644 to 640x640, which is the size of the image
        downsampled_logits = torch.nn.functional.interpolate(outputs.logits,
                                                   size=[640,640],
                                                   mode="bilinear", align_corners=False)
        predicted_map = downsampled_logits.argmax(dim=1)
        results=predicted_map.squeeze().cpu().numpy()
        
        # Calculate FN and FP
        false_negatives = np.sum((results == 0) & (ground_truth[0] == 1))
        false_positives = np.sum((results == 1) & (ground_truth[0] == 0))
        
        # Total number of instances
        total_instances = np.prod(results.shape)
        
        # Calculate percentages
        percentage_fn = (false_negatives / total_instances) 
        percentage_fp = (false_positives / total_instances) 
        
        metrics = self.train_mean_iou._compute(
            predictions=results,
            references=ground_truth[0],
            num_labels=self.num_classes,
            ignore_index=254,
            reduce_labels=False,
        )
        # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        
        # Re-define metrics dict to include per-category metrics directly
        metrics = {
            'loss': loss, 
            "mean_iou": metrics["mean_iou"], 
            "mean_accuracy": metrics["mean_accuracy"],
            "False Negative": percentage_fn,
            "False Positive": percentage_fp,
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k,v in metrics.items():
            self.log(k,v,sync_dist=True, batch_size=config.BATCH_SIZE)
        return(metrics)
        
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)
