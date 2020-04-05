# Noiseless
Project for denoising images

## Dataset decription

For the purposes of current project [TAMPERE IMAGE DATABASE 2008]( http://www.ponomarenko.info/tid2008.htm ) was chosen. It has 25 reference images (clean) and 1700 corrupted images (each reference image is corrupted with 17 types of noise, each noise has 4 levels of strength).

### How to download dataset

In order to obtain the dataset, one can use this [link to download it (550 mb)](http://www.ponomarenko.info/tid/tid2008.rar).

### How to add our custom noise

To expand the set of distorted images you can apply our custom gaussian noise or random erasing noise. For this purpose we have ```gaussian_noise_overlay.py``` and ```erasing_noise_overlay.py```  in the folder ```./scripts/```.

Example of usage:

```
python gaussian_noise_overlay.py --path_to_clean_images=<str> --path_to_corrupted_images=<str> [--noise_type=<int>]
```

```
python erasing_noise_overlay.py --path_to_clean_images=<str> --path_to_corrupted_images=<str> [--noise_type=<int>]
```



### How to create folders structure necessary for training our model

To train a model we need to have the following data structure:

```
data
│   ├───clean
│   ├───noised_{i}
│   └───only_noise_{i}
```

```data/```

- ```clean/``` - folder with reference images
- ```noised_{i}/``` -  folder with noised images, where "i" is the type of noise (1 to 19 in case you use 17 noises from the initial dataset and 2 our custom noises)
- ```only_noise_{i}/``` - folder with difference of noised images and reference images, where "i" has the same meaning as above

Such structure can be generated via our script ```data_splitting.py```  in the folder ```./scripts/```.

Example usage:

```
python data_splitting.py --path_to_corrupted=<str> --path_to_reference=<str> 
--path_for_processed=<str> [--num_noises=<int> --noise_level=<int>]
```


## Model installation and usgae

### Installation

conda smth


### How to train model

In order to train model you should run `train_test.py` script.

In order to get full description of flags one can use the command:

```
python train_test.py -h
```

The example of usage:

```

python ./train_test.py --train=True --noise_types="1, 2" --image_size="512, 384" --frame_size="64, 64" --overlay_size="5, 5" --latent_clean_size=0.9 --batch_size=4 --epochs=20 --test=True
```
