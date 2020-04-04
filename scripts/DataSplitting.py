import sys, getopt
import cv2 as cv
import numpy as np
import os

# from pathlib import Path
from tqdm import tqdm

NUM_NOISES = 17
NOISE_LEVEL = 4


def arguments_parsing(argv):
    path_to_corrupted = ""
    path_to_reference = ""
    path_for_processed = ""
    num_noises = NUM_NOISES
    noise_level = NOISE_LEVEL
    try:
        opts, args = getopt.getopt(argv, "h", ["path_to_corrupted=", "path_to_reference=",
                                               "path_for_processed=", "num_noises=", "noise_level="])
    except getopt.GetoptError:
        print("./DataSplitting.py --path_to_corrupted=<str>"+
              "--path_to_reference=<str> --path_for_processed=<str> [--num_noises=<int> "+
              "--noise_level=<int>]")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print("./DataSplitting.py --path_to_corrupted=<str>"+
              "--path_to_reference=<str> --path_for_processed=<str> [--num_noises=<int> "+
              "--noise_level=<int>]")
            sys.exit()
        elif opt == "--path_to_corrupted":
            path_to_corrupted = os.path.join(arg)
        elif opt == "--path_to_reference":
            path_to_reference = os.path.join(arg)
        elif opt == "--path_for_processed":
            path_for_processed = os.path.join(arg)
        elif opt == "--num_noises":
            num_noises = int(arg)
        elif opt == "--noise_level":
            noise_level = int(arg)
            
    return path_to_corrupted, path_to_reference, path_for_processed, num_noises, noise_level

def ProcessDataset(path_to_corrupted, path_to_reference, path_for_processed, num_noises=17, noise_level=4):
    """
    Process dataset: create folders structure; convert all images to PNG;
    create noise_only images; split images into corresponding subfolders.
    """
    path_noised = os.path.join(path_for_processed, "noised/")
    path_clean = os.path.join(path_for_processed, "clean/")

    # Path(path_clean).mkdir(parents=True, exist_ok=True)
    os.makedirs(path_clean, exist_ok=True)
    for i in range(1, num_noises + 1):
            # Path(f"{path_noised[:-1]}_{i}/").mkdir(parents=True, exist_ok=True)
            os.makedirs(f"{path_noised[:-1]}_{i}/", exist_ok=True)
    for noised_name in os.listdir(path_to_corrupted):
        noise_type = noised_name.split("_")[1]
        if int(noise_type[0]) == 0:
            noise_type = noise_type[1:]
        current_noise_level = noised_name.split("_")[2][:-4]
        noised = cv.imread(path_to_corrupted + noised_name)
        if int(current_noise_level) == 4:
            cv.imwrite(f"{path_noised[:-1]}_{noise_type}/" + noised_name[:-4] + ".PNG", np.int32(noised))
    for clean_name in os.listdir(path_to_reference):
        clean = cv.imread(path_to_reference + clean_name)
        cv.imwrite(path_clean + clean_name[:-4] + ".PNG", np.int32(clean))

    def CreateNoise(path_noised, path_clean, path_only_noise):
        listdir_noised = os.listdir(path_noised)
        listdir_clean = os.listdir(path_clean)
        for idx, noised_name in enumerate(listdir_noised):
            noised = np.int32(cv.imread(path_noised + noised_name))
            clean = np.int32(cv.imread(path_clean + listdir_clean[idx]))
            noise = noised - clean
            cv.imwrite(path_only_noise + "noise" + noised_name[-11:-4] + ".PNG", noise)

    for i in tqdm(range(1, num_noises + 1)):
        path_noised = f"{path_for_processed}noised_{i}/"
        path_only_noise = path_for_processed + f"only_noise_{i}/"
        # Path(path_only_noise).mkdir(parents=True, exist_ok=True)
        os.makedirs(path_only_noise, exist_ok=True)
        CreateNoise(path_noised, path_clean, path_only_noise)


def main(argv):
    path_to_corrupted, path_to_reference, path_for_processed, num_noises, noise_level = arguments_parsing(argv)
    ProcessDataset(path_to_corrupted, path_to_reference, path_for_processed, num_noises, noise_level)

if __name__ == "__main__":
    main(sys.argv[1:])