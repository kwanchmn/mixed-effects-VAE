This repository contains all the codes used for implementing mixed-effects Variational Auto-encoder (VAE) in my thesis for the Master of Science in Statistics degree at the University of Geneva.

Below is a brief description of the important folders and files in this repository: 
- metadata: a folder containing csv files with the filenames, states, subject IDs and other attributes of the images in the MRL eye dataset.
- mrleye_classification: a folder with the trained feed-forward neural network models, their training histories, and accuracy scores on the test data for the classification task using encodings from mixed-effects VAE model as inputs.
- mvae_fixeff: a folder containing the trained neural network models for the fixed-effects part of the mixed-effects VAE models with different values of $\sigma^2_h$
- mvae_raneff: a folder containing the trained `brms` random-effects models for mixed-effects VAE models with different values of $\sigma^2_h$
- data_explore: a `python` script with exploratory data analysis of the MRL eye dataset, can be used to download all the images from the `Kaggle` API
- data_processing: a `python` script for pre-processing the dataset to be used for implementing mixed-effects VAE
- mvae_script: an `R` script implementing mixed-effects VAE, with the MRL eye dataset as the empirical data for application
  