import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import kagglehub
import keras

# # Download latest version
# path = kagglehub.dataset_download("akashshingha850/mrl-eye-dataset/versions/4")
# print("Path to dataset files:", path)

mrl_train = keras.utils.image_dataset_from_directory(
    "mrl-eye-dataset/versions/4/data/train", 
    image_size = (64, 64),  # since original sizes are different
    shuffle = False, 
    color_mode="rgb"
)

"""
Getting metadata of data sets
"""

def extract_metadata(awake_dir, sleepy_dir):

    awake_dir_list = os.listdir(awake_dir)
    sleepy_dir_list = os.listdir(sleepy_dir)
    awake_df = pd.DataFrame(zip(awake_dir_list, ["awake"] * len(awake_dir_list)))
    sleepy_df = pd.DataFrame(zip(sleepy_dir_list, ["sleepy"] * len(sleepy_dir_list)))

    images_df = pd.concat([awake_df, sleepy_df])
    images_df.columns = ["filename", "state"]

    # Getting metadata based on file name
    metadata = images_df["filename"].str.replace(".png", "").str.split("_", expand=True)
    metadata.columns = ["subject", "index", "eye_state", "gender", 
                        "glasses", "reflections", "lighting", "sensor_id"]
    metadata["subject"] = metadata["subject"].str.replace(r"^s\d{2}", "", regex=True).astype(int)

    # Concatenating to the main DataFrame
    images_df = pd.concat([images_df, metadata], axis = 1).reset_index(drop=True)
    return images_df

# Training set
train_awake_dir = "images/train/awake"
train_sleepy_dir = "images/train/sleepy"
train_df = extract_metadata(train_awake_dir, train_sleepy_dir)
# print(train_df.head())

# Validation set
val_awake_dir = "images/val/awake"
val_sleepy_dir = "images/val/sleepy"
val_df = extract_metadata(val_awake_dir, val_sleepy_dir)
# print(val_df.head())

# And test set
test_awake_dir = "images/test/awake"
test_sleepy_dir = "images/test/sleepy"
test_df = extract_metadata(test_awake_dir, test_sleepy_dir)
# print(test_df.head())

"""
Inspecting some images
"""

def plot_images(data, folder_path, seed = 123):
    np.random.seed(seed)
    img_samps = np.random.choice(data.shape[0], size = 6)

    fig, axes = plt.subplots(2, 3, figsize = (10, 6))
    for i, ax in enumerate(axes.flatten()): 
        idx = img_samps[i]
        img_dir = f"{folder_path}/{data["state"][idx]}/{data["filename"][idx]}"
        eye_img = plt.imread(img_dir)
        ax.set_title(data["filename"][idx] )
        ax.imshow(eye_img, cmap = "gray")

    plt.tight_layout()
    plt.show()

plot_images(train_df, "images/train")

"""
Resizing images to the same dimensions for inputs to VAE
"""

def resize_image(data, file_list, org_path, dest_path, 
                 new_size = (64, 64)):
    
    try:
        os.mkdir(dest_path)
    except FileExistsError:
        pass

    for idx, file in enumerate(file_list):
        img_dir = f"{org_path}/{data['state'][idx]}/{file}"
        with Image.open(img_dir) as image:
            new_image = image.resize(new_size)
            new_image.save(f"{dest_path}/{file}")
    
    print(f"Finished resizing {idx + 1} images into {new_size}!")
    
# # Resizing into 64 x 64
# resize_image(train_df, train_df["filename"], "images/train", "images/train_resize")
# resize_image(val_df, val_df["filename"], "images/val", "images/val_resize")
# resize_image(test_df, test_df["filename"], "images/test", "images/test_resize")

"""
Exploratory Data Analysis
"""
# 1. How many images per subject?
train_df["subject"].value_counts()
val_df["subject"].value_counts(normalize=True)
test_df["subject"].value_counts(normalize=True)
# Same proportion for each subject in three data sets

# 2. Number of images by subject and eye state?
train_df.groupby(["subject", "state"])["subject"].value_counts()
val_df.groupby(["subject", "state"])["subject"].value_counts()
test_df.groupby(["subject", "state"])["subject"].value_counts()
# Also looks equally spread among three data sets

"""
Saving metadata files as csv
"""
# train_df.to_csv("metadata/train_img_metadata.csv")
# val_df.to_csv("metadata/val_img_metadata.csv")
# test_df.to_csv("metadata/test_img_metadata.csv")
