import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

"""Fetching the dataset"""

# # Download latest version
# path = kagglehub.dataset_download("akashshingha850/mrl-eye-dataset/versions/4")
# print("Path to dataset files:", path)

"""
Getting metadata of data sets
"""
def extract_metadata(awake_dir, sleepy_dir):
    
    # Directory
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
    images_df = pd.concat([images_df, metadata[["subject", "index"]]], axis = 1).reset_index(drop=True)
    return images_df

# Training set
train_awake_dir = r"mrl-eye-dataset\versions\4\data\train\awake"
train_sleepy_dir = r"mrl-eye-dataset\versions\4\data\train\sleepy"
train_df = extract_metadata(train_awake_dir, train_sleepy_dir)
print(train_df.head())

# Validation set
val_awake_dir = r"mrl-eye-dataset\versions\4\data\val\awake"
val_sleepy_dir = r"mrl-eye-dataset\versions\4\data\val\sleepy"
val_df = extract_metadata(val_awake_dir, val_sleepy_dir)
print(val_df.head())

# And test set
test_awake_dir = r"mrl-eye-dataset\versions\4\data\test\awake"
test_sleepy_dir = r"mrl-eye-dataset\versions\4\data\test\sleepy"
test_df = extract_metadata(test_awake_dir, test_sleepy_dir)
print(test_df.head())

"""
Inspecting some images in training set
"""

def read_images(folder_path, img_name):
    img_dir = f"{folder_path}/{img_name}"
    return plt.imread(img_dir)

def plot_images(data, folder_path, seed = 1, nrow = 3, ncol = 3):
    np.random.seed(seed)
    img_samps = np.random.choice(data.shape[0], size = nrow*ncol)

    fig, axes = plt.subplots(nrow, ncol, figsize = (10, 6))
    for i, ax in enumerate(axes.flatten()): 
        idx = img_samps[i]
        img_dir = f"{folder_path}/{data["state"][idx]}/{data["filename"][idx]}"
        eye_img = plt.imread(img_dir)
        ax.set_title(data["filename"][idx].replace(".png", ""))
        ax.imshow(eye_img, cmap = "gray")

    plt.tight_layout()
    plt.show()

plot_images(train_df, r"mrl-eye-dataset\versions\4\data\train")

"""
Exploratory Data Analysis
"""
train_df.info()
val_df.info()
test_df.info()

# 1. How many unique clusters (i.e. subjects) do we have in the data?
len(train_df['subject'].unique())

# Same proportion for each subject in three data sets
train_df["subject"].value_counts()  # minimum n_i = 244
val_df["subject"].value_counts()

# 2. How is class (im)balance for state?

# Very balanced, basically 50/50 split, nice
train_df["state"].value_counts(normalize=True)
val_df["state"].value_counts(normalize=True)

# 2. Number of images by subject and eye state?
# Also looks equally spread among three data sets
subject_split = train_df.groupby(["subject", "state"])["subject"].value_counts()
print(subject_split)

# ===============
# Obsolete codes below...
# ===============
# """ 
# Getting a subset of training data images
# """
# def sample_train_subjects(subject_list, num_clusters, seed = 123): 
#     
#     np.random.seed(seed)
#     unique_subjects = np.unique(subject_list)
#     training_subjects = np.random.choice(unique_subjects, 
#                                          size = num_clusters, replace = False)
#     test_subjects = np.setdiff1d(unique_subjects, training_subjects)
#     return {"training": training_subjects, "test": test_subjects}
# 
# subject_groups = sample_train_subjects(train_df["subject"], 30)
# train_subjects = subject_groups['training']
# 
# """
# Down-sampling the original dataset for faster training
# """
# def data_subset(data_df, subject_set, cluster_samp_n, seed = 123):
#     
#     df_subset = data_df.query("subject in @subject_set")
#     
#     if cluster_samp_n is None:
#         subject_counts = df_subset["subject"].value_counts()
#         cluster_samp_n = subject_counts.min()
#         
#     sub_sample = df_subset.groupby("subject").sample(n = cluster_samp_n, random_state = seed)
#     return sub_sample
# 
# # Test
# cluster_sub_n = 100
# train_subset = data_subset(train_df, train_subjects, cluster_sub_n)
# 
# # Checking if the subset is too different from the rest of training data
# train_subset['state'].value_counts()  # 50-50 split between two states
# train_subset.groupby(["subject", "state"]).state.value_counts()
# 
# """
# Sampling hidden variables according to distributions
# """
# 
# def hidden_var_sampler(cluster_n, cluster_cov, hidden_dim, hidden_samp_n = None):
#     
#     if cluster_cov.shape[0] != hidden_dim:
#         ValueError("Incompatible hidden variable dimension with the cluster covariance structure!")
#     
#     # Building the covariance matrix
#     hidden_cov_mat = np.ones((cluster_n, cluster_n))
#     hidden_cov_mat = np.kron(hidden_cov_mat, cluster_cov)
#     np.fill_diagonal(hidden_cov_mat, 1)
#                 
#     # Sampling from an MVN with mean 0 and the above covariance matrix
#     norm_mean = np.zeros(shape = cluster_n * hidden_dim) 
#     hidden_var_samp = np.random.multivariate_normal(mean = norm_mean, 
#                                                     cov = hidden_cov_mat, 
#                                                     size = hidden_samp_n)
#     return hidden_var_samp
#     
# # Test
# cluster_cov_mat = np.array([.5, 0, 0, .5]).reshape(2, 2)
# test_mat = hidden_var_sampler(cluster_n = 10, cluster_cov = cluster_cov_mat, 
#                               hidden_dim = 2, hidden_samp_n = 100)
# 
# samp_mean = np.mean(test_mat, axis = 1).round(2)
# samp_cov = np.cov(test_mat).round(2)
# print(test_mat.shape)
# 
# """
# Duplicating cluster observations for the sampled hidden variables
# """
# 
# def hidden_var_df_generator(subject_set, cluster_n, cluster_cov, hidden_dim, hidden_samp_n = None):
#     hidden_var_df = pd.DataFrame()
#     
#     for subject in subject_set:
#         subject_id = pd.Series([subject] * cluster_n)
#         subject_hidden_samp = pd.DataFrame(hidden_var_sampler(cluster_n, cluster_cov, 
#                                                               hidden_dim, hidden_samp_n))
#         subj_hidden_df = pd.concat([subject_id, subject_hidden_samp], axis = 1)
#         hidden_var_df = pd.concat([hidden_var_df, subj_hidden_df], axis = 0)
#     
#     return hidden_var_df
#         
# seed = 123
# cluster_sub_n = 100
# np.random.seed(seed)
# subset_hidden_vars = hidden_var_df_generator(train_subjects, cluster_sub_n, cluster_cov_mat, 
#                                              hidden_dim = 2, hidden_samp_n = 100)
# 
# hidden_dim_labels = []
# 
# for r in range(1, cluster_sub_n + 1):
#     for j in range(1, 3):
#         hidden_dim_labels.append(f"h_i{j}^{r}")
# 
# subset_hidden_vars.columns = ["subject"] + hidden_dim_labels
# 
# 
# 
# """
# Making them into dataframes for input and output
# """
# train_subset.to_csv("eye_train_subset.csv")
# 
# """
# Resizing images to the same dimensions for inputs to VAE
# """

# mrl_train = keras.utils.image_dataset_from_directory(
#     "mrl-eye-dataset/versions/4/data/train", 
#     image_size = (64, 64),  # since original sizes are different
#     shuffle = False, 
#     color_mode="rgb"
# )

# def resize_image(data, file_list, org_path, dest_path, 
#                  new_size = (64, 64)):
    
#     try:
#         os.mkdir(dest_path)
#     except FileExistsError:
#         pass

#     for idx, file in enumerate(file_list):
#         img_dir = f"{org_path}/{data['state'][idx]}/{file}"
#         with Image.open(img_dir) as image:
#             new_image = image.resize(new_size)
#             new_image.save(f"{dest_path}/{file}")
    
#     print(f"Finished resizing {idx + 1} images into {new_size}!")
    
# # Resizing into 64 x 64
# resize_image(train_df, train_df["filename"], "images/train", "images/train_resize")
# resize_image(val_df, val_df["filename"], "images/val", "images/val_resize")
# resize_image(test_df, test_df["filename"], "images/test", "images/test_resize")
