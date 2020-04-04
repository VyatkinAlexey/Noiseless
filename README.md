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

## How to train model

In order to train model you should run `train_test.py` script.

In order to get full description of flags one can use the command:

```
python train_test.py -h
```

The example of usage:

```
python ./train_test.py --train=True --noise_folders='noised_1' --image_size='512, 384' --frame_size='64, 64' --overlay_size='5, 5' --latent_clean_size=0.9 --batch_size=4 --epochs=20 --test=True
```
