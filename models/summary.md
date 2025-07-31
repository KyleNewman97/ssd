# Summary

## Info

Run ID: `49ffe3fa-f06f-4b7d-9677-2fa87f112544`

Epochs: `12`

## Overview

A simple SSD model trained on the COCO dataset. No augmentation was used.

## Results

Training was paused at epoch 12 - the model hadn't plateaued. Further training could improve accuracy.

The model was run over the validation dataset and then visualised. The three many observations where:

1. Many of the objects would have more than one bounding box associated with them.
2. Boxes were in generally the right location, but poorly fit the objects.
3. The model struggled to detect smaller objects.

## Conclusions

To further improve the model I would recommend:

1. Adjust the size of anchor boxes - the distribution of anchors sizes and ground truth sizes are vastly different. I believe this is why smaller objects are not being detected.
2. Adjusting the anchor box to ground truth association algorithm. I believe too many anchors are getting associated with each object - meaning even far away boxes have to try to be transformed into the correct positions.
