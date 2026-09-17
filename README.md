# Task 05: Neural Style Transfer (NST)

This repository contains an implementation of **Neural Style Transfer** built in PyTorch using a pre-trained **VGG19** feature extractor (Gatys et al.).

## Project Overview
Neural Style Transfer takes a content image and a style image, optimizing a generated output image to preserve the high-level content structure of the target while adopting the artistic texture and color palette of the style image.

## Project Structure
- `task5_style_transfer.py`: Script defining image preprocessing, feature extraction via VGG19, and optimization pipeline.

## Requirements
```bash
pip install torch torchvision pillow
