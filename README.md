# Noiseless
Project for denoising images

## Dataset decription

For the purposes of current project [Natural Image Noise Dataset](https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset) was chosen.

### How to download dataset

In order to obtain dataset (which is free, non-commercial), one can use `scripts/download.py`,
which is the property of aforementioned project.

The example of usage:

```
python download.py --use_wget --target_dir DESTINATION_DIR
```

In order to get full description of flags one can use the command:

```
python download.py -h
```
