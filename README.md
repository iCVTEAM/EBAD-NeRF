# Code for Deblurring Neural Radiance Fields with Event-driven Bundle Adjustment (ACM MM 2024)
This is an official PyTorch implementation of the EBAD-NeRF. Click [here](https://icvteam.github.io/EBAD-NeRF.html) to see the video and supplementary materials in our project website.

## Code

### Synthetic Data

The configs of the synthetic data are in the config_blender.txt file. Please download the synthetic data below and put it into the corresponding file (./data/blender_llff/). Then you can use the command below to train the model.

```
python train_blender.py --config config_blender.txt
```

## Datasets

### Synthetic Data

The synthetic data can be downloaded at [here](https://drive.google.com/drive/folders/112SGk_v-fxaUKz7w9dOqhXZRnkwTMkIt?usp=sharing). We use five Blender scenes from BAD-NeRF to construct this dataset. To increase the difficulty of the data, we add non-uniform camera shake. As shown in the folder, each scene folder contains five parts: 

"images": images for training.

"images_gt_blur": ground truth images of blur view for testing.

"images_gt_novel": ground truth images of novel view for testing.

"events.pt": the event data for training.

"pose_bounds.npy": the initial poses for training.
