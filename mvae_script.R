# ===============================
# Pre-requisites
# ===============================

# Packages
library(keras3)
library(knitr)
library(brms)
library(tidyverse)
library(MVN)
library(MASS)
library(docstring)
library(imager)
library(reticulate)
setwd("C:/Users/kenji/OneDrive/Desktop/Thesis/Code/mrl_eye_data")

# Ensuring reproducibility of the results
seed = 114514
set.seed(seed)

# Metadata of the images
mrl_train_df = read_csv("metadata/train_img_metadata.csv", col_select = -1)
mrl_val_df = read_csv("metadata/val_img_metadata.csv", col_select = -1)
mrl_test_df = read_csv("metadata/test_img_metadata.csv", col_select = -1)

# ===============================
# Randomly selecting clusters to be in training data
# ===============================

known_clusters_sampler = function(cluster_col, known_cluster_n) {
  
  # Description:
  #   Randomly selects the clusters to be included in the training data given a column indicating the grouping factor
  
  # Arguments:
  #   cluster_col: a column indicating the grouping factor in a dataframe
  #   known_cluster_n: number of clusters to be included in the training data, 
  #                    must be equal to or less than the total number of clusters
  
  # Return:
  #   cluster_list: a list containing vectors of clusters included ($known) 
  #   and excluded ($unknown) in the training data
  
  unique_clusters = unique(cluster_col)
  known_clusters= sample(unique_clusters, known_cluster_n)
  unknown_clusters = setdiff(unique_clusters, known_clusters)
  
  cluster_list = list(known = known_clusters, unknown = unknown_clusters)
  
  return(cluster_list)
}

# Sampling clusters in training data
known_clusters_n = 30
cluster_split = known_clusters_sampler(mrl_train_df[["subject"]], known_clusters_n)

# Sub-setting training data with above cluster list
mrl_train_df_subset = mrl_train_df %>% 
  filter(subject %in% cluster_split$known)

# ===============================
# Down-sampling original MRL eye dataset
# ===============================

data_downsampler = function(data, cluster_col, cluster_size) {
  
  # Description:
  #   Randomly selects a subset of the original dataset 
  
  # Arguments:
  #   data: the dataframe to be downsized
  #   cluster_col: the grouping factor of the clusters
  #   cluster_size: the number of observations to be sampled for each cluster
  
  # Return:
  #   data_subset: the downsized dataframe sub-sampled for each cluster with the desired number of observations 
  
  data_subset = data %>% 
    group_by(across(cluster_col)) %>% 
    slice_sample(n = cluster_size)
  
  return(data_subset)
}

# Downsizing subset mrl train data
cluster_obs_n = 100
mrl_train_downsized = data_downsampler(mrl_train_df_subset, "subject", cluster_obs_n)
mrl_train_downsized = subset(mrl_train_downsized, select = c("filename", "state", "subject"))

# Full filepath
create_full_filepath = function(dataframe, dataset_type) {
  
  # Description:
  #   Creates the full filepaths for the MRL eye dataset images
  
  # Arguments:
  #   dataframe: the dataframe containing the metadata of the images
  #   dataset_type: whether the dataframe is the training ("train"), validation ("val") or test ("test") data
  
  # Return:
  #   dataframe: the original dataframe with the `filename` column being replaced with the full file path
  
  
  filepath_list = paste0(r"{mrl-eye-dataset\versions\4\data\}",
                         dataset_type, 
                         r"{\}",
                         dataframe$state, 
                         r"{\}", 
                         dataframe$filename)
  dataframe$filename = filepath_list
  dataframe
}

mrl_train_downsized = create_full_filepath(mrl_train_downsized, "train")

# ===============================
# Creating latent variables sampling function
# ===============================

latent_h_sampler = function(latent_dim, cluster_obs_n, sigma2_h) {
  
  # Description:
  #   Samples latent variables from a multivariate Gaussian distribution with the input covariance matrix
  
  # Arguments:
  #   latent_dim: the dimension of the latent variable vector
  #   cluster_obs_n: the number of observations in a cluster
  #   sigma2_h: the element-wise covariance of the same dimension of any two different latent variables in cluster i
  
  # Return:
  #   A matrix with dimensions of cluster_obs_n x latent_dim containing the sample from the MVN for cluster i
  
  # Covariance of cluster i
  within_cluster_cov = diag(sigma2_h, latent_dim)
  cov_mat_h_i = matrix(rep(1, cluster_obs_n**2), nrow = cluster_obs_n, byrow = TRUE) %x% within_cluster_cov
  diag(cov_mat_h_i) = 1
  
  mean_h_i = rep(0, times = latent_dim * cluster_obs_n) 
  
  # Sampling h cluster-wise
  h_i_samp = mvrnorm(mu = mean_h_i, Sigma = cov_mat_h_i)
  return(matrix(h_i_samp, nrow = cluster_obs_n, ncol = latent_dim, byrow = TRUE))
}

# ===============================
# # Testing if sample converges
# ===============================
# latent_samp_R = 1e4
# 
# # Drawing samples of h_i
# samples_list <- replicate(latent_samp_R, {
#   latent_h_sampler(latent_dim, cluster_obs_n, sigma2_h)
# }, simplify = FALSE)
# 
# # Checking if the sample mean and covariance matrix are similar to population one
# col_mean = Reduce("+", samples_list) / latent_samp_R
# mean_h_i = rep(0, times = latent_dim * cluster_obs_n)
# max(abs(col_mean - mean_h_i))
# 
# # And the covariance matrix
# # Flatten each 100x2 sample back into a 200-length vector using byrow = TRUE
# samples_matrix <- t(sapply(samples_list, function(mat) as.vector(t(mat))))
# 
# # Compute the 200x200 sample covariance matrix
# within_cluster_cov = diag(sigma2_h, latent_dim)
# cov_mat_h_i = matrix(rep(1, cluster_obs_n**2), nrow = cluster_obs_n, byrow = TRUE) %x% within_cluster_cov
# sample_cov <- cov(samples_matrix)
# max(abs(sample_cov - cov_mat_h_i))

# ===============================
# Formatting simulated latent variables as pseudo-response 
# ===============================

generate_response_h = function(dataframe, latent_samp_R, 
                               cluster_num, cluster_obs_n, 
                               sigma2_h, latent_dim = 2) {
  
  # Description:
  #   Generates latent variables as pseudo-responses for the observations in the input dataframe
  
  # Arguments:
  #   dataframe: the dataframe containing the observations in the training data
  #   latent_samp_R: the number of samples of latent variables to be generated for each observation
  #   cluster_num: the number of clusters of the grouping factor
  #   cluster_obs_n: the number of observations in a cluster
  #   sigma2_h: the element-wise covariance of the same dimension of any two different latent variables in cluster i
  #   latent_dim: the dimension of the latent variable vector
  
  # Return:
  #   A dataframe with the latent variable samples attached to the observation as pseudo-responses, 
  #   where each unique observation will be duplicated latent_samp_R times 
  #   for the corresponding latent variable sample
  
  new_df = dataframe
  
  for (r in seq_len(latent_samp_R)) {
    
    latent_h_samp = replicate(cluster_num, {
      latent_h_sampler(latent_dim, cluster_obs_n, sigma2_h)
    }, simplify = FALSE)
    
    latent_h_samp = do.call(rbind, latent_h_samp)
    
    latent_h_samp = lapply(split(latent_h_samp, seq_len(nrow(latent_h_samp))), as.numeric)
    
    h_col_name = paste0("h_", r)
    
    new_df[[h_col_name]] = latent_h_samp
  }
  
  # Long format
  last_h = paste0("h_", latent_samp_R)
  new_df = pivot_longer(new_df, 
                        cols = "h_1":last_h, 
                        names_to = "latent_h", 
                        values_to = "value")
  return(new_df)
}

# ===============================
# Training with different sigma2_h
# ===============================

# Loading images in batches for memory
image_batch_generator <- function(df, batch_size, target_size = c(64, 64)) {
  
  # Description:
  #   A generator function for loading the images into batches to train keras models
  
  # Arguments:
  #   df: the dataframe containing all the observations
  #   batch_size: number of images to be temporarily loaded into each batch
  #   target_size: the height and width of each image
  
  # Return:
  #   NULL
  
  i <- 1
  N <- nrow(df)
  df_shuffled <- df[sample(N), ]
  
  function() {
    if (i > N) {
      df_shuffled <<- df[sample(N), ]
      i <<- 1
    }
    
    batch_end <- min(i + batch_size - 1, N)
    batch_df <- df_shuffled[i:batch_end, ]
    actual_batch_size <- nrow(batch_df)
    
    # Input images
    x_batch <- array(0, dim = c(actual_batch_size, target_size[1], target_size[2], 3))
    
    # Regression targets: convert list of 2D vectors into a matrix
    y_batch <- do.call(rbind, batch_df$value)
    
    for (j in seq_len(actual_batch_size)) {
      img <- image_load(batch_df$filename[j], target_size = target_size)
      x_batch[j,,,] <- image_to_array(img) / 255
    }
    
    i <<- batch_end + 1
    list(x_batch, y_batch)
  }
}

# Creating a validation set
mrl_val_df_subset = mrl_val_df %>% 
  filter(subject %in% cluster_split$known)

val_cluster_obs_n = 30
mrl_val_df_downsized = data_downsampler(mrl_val_df_subset, "subject", val_cluster_obs_n)
mrl_val_df_downsized = subset(mrl_val_df_downsized, select = c("filename", "state", "subject"))
mrl_val_df_downsized = create_full_filepath(mrl_val_df_downsized, "val")

# Latent variable parameters
latent_dim = 2
latent_samp_R = 100
input_size = c(64, 64, 3)
sigma2_h_list = c(0, 0.3, 0.5, 0.7)  # Hyper-parameter

# NN history and batch sizes
batch_size = 3000
nn_encoder_history_list = list()

for (sigma2_h in sigma2_h_list) {
  
  # No redundant training if new sigma2_hs are considered in the future
  keras_name = paste0("mvae_fixeff/nn_with_sigma2h_", sigma2_h, ".keras")
  if (file.exists(keras_name)) {
    next
  }
  
  # Generating latent variables
  mrl_train_with_h = generate_response_h(mrl_train_downsized, latent_samp_R, known_clusters_n, cluster_obs_n, sigma2_h)
  mrl_val_with_h = generate_response_h(mrl_val_df_downsized, latent_samp_R, known_clusters_n, val_cluster_obs_n, sigma2_h)
  
  # Batching datasets
  train_generator = image_batch_generator(mrl_train_with_h, batch_size)
  val_generator = image_batch_generator(mrl_val_with_h, batch_size)
  
  # Early stopping for neural network
  callbacks_list = list(callback_early_stopping(monitor="val_loss",
                                                patience = 3,  
                                                restore_best_weights = TRUE, 
                                                min_delta = 1e-3))
  
  # Building the neural network
  resnet = application_resnet50_v2(include_top = FALSE,
                                   weights = 'imagenet',
                                   input_shape = input_size)
  resnet$trainable = FALSE
  
  # Setting up layers up to the latent variables as output
  flatten_layer = layer_flatten()
  flatten_layer$build(shape(NULL,  2, 2, 2048))
  
  ## Add a drop-out layer, and l1 penalty
  dropout_rate = 0.5
  dropout_layer = layer_dropout(rate = dropout_rate)
  
  output_layer = layer_dense(units = 2)
  output_layer$build(shape(NULL,  2 * 2 * 2048))
  
  # Building the sequential model
  model = keras_model_sequential(input_size)
  model %>%
    resnet %>%
    flatten_layer %>%
    dropout_layer %>%
    output_layer
  
  model %>% compile(optimizer = "adam",
                    loss = 'mse')
  
  # Fitting neural network for fixed-effects part
  nn_hist = model %>% fit(
    x = train_generator,
    steps_per_epoch = ceiling(nrow(mrl_train_with_h) / batch_size),
    validation_data = val_generator,
    validation_steps = ceiling(nrow(mrl_val_with_h) / batch_size),
    epochs = 50,
    callbacks = callbacks_list
  )
  
  # Saving the model for later
  save_model(model, paste0("mvae_fixeff/nn_with_sigma2h_", sigma2_h, ".keras"))
  
  nn_encoder_history_list[[as.character(sigma2_h)]] = nn_hist
}

# # Reading training history of encoder NNs
# saveRDS(nn_encoder_history_list, file = "mvae_fixeff/nn_encoder_history_list.rds")
nn_encoder_history_list = readRDS("mvae_fixeff/nn_encoder_history_list.rds")

# ===============================
# Plotting training history 
# ===============================

# Function to plot each subplot
plot_single_loss <- function(model_hist, sigma2_h, title = NULL) {
  
  # Description:
  #   Plots the evolution of the training and validation losses of a keras model 
  #   as a function of training epochs
  
  # Arguments:
  #   model_hist: the training history of a keras model 
  #   sigma2_h: the element-wise covariance of the mixed-effects VAE model
  #   title: title of the plot, defaults to the value of sigma2_h used
  
  # Return:
  #   A base R plot object
  
  n_epochs <- length(model_hist$metrics$loss)
  epochs = seq_len(n_epochs)
  loss <- model_hist$metrics$loss
  val_loss <- model_hist$metrics$val_loss
  y_range <- range(c(loss, val_loss))
  
  # Default title
  if (is.null(title)) {
    title = bquote(sigma[h]^2 == .(sigma2_h))
  }
  
  plot(epochs, loss, type = "n",
       ylim = y_range,
       xlab = "Epoch", ylab = "Loss",
       main = title,
       cex.main = 1.4, cex.axis = 1.1, 
       cex.lab = 1.2, axes = FALSE)
  
  # Custom integer x-axis
  x_ticks <- pretty(epochs, n = n_epochs)
  x_ticks <- x_ticks[x_ticks %% 1 == 0]
  axis(1, at = x_ticks, labels = x_ticks)
  axis(2)
  box()
  
  # Add loss and val_loss lines with transparency and markers
  lines(epochs, loss, type = "o",
        col = adjustcolor("#1f77b4", alpha.f = 0.7),
        lwd = 1.5, pch = 16)
  
  lines(epochs, val_loss, type = "o",
        col = adjustcolor("#ff7f0e", alpha.f = 0.7),
        lwd = 1.5, pch = 17, lty = 2)
}

# Function to plot 2x2 subplots of 4 sigma2_h values at once
plot_all_losses = function(model_hist_list, sigma2_h_list, global_title) {
  
  # Description:
  #   Plots a list of training histories of keras models 
  
  # Arguments:
  #   model_hist_list: a list of keras models training histories
  #   sigma2_h_list: a list of values of sigma2_h corresponding to the list of keras models training histories
  #   global_title: the global title of all the subplots of keras model histories
  
  # Return:
  #   A 2x2 grid of subplots plotting all the model training histories 
  #   of the 4 values of sigma2_h considered in the manuscript
  
  # Reserve outer margin space for the global title (top = 4 lines)
  par(oma = c(0, 0, 4, 0))  # bottom, left, top, right
  
  # Create layout: title (mtext), 2x2 panels, bottom legend
  layout_matrix <- matrix(c(
    1, 2,
    3, 4,
    5, 5
  ), nrow = 3, byrow = TRUE)
  
  layout(layout_matrix, heights = c(1, 1, 0.25))
  
  # Draw the 4 subplots
  par(mar = c(4, 4, 3, 1))  # standard subplot margins
  for (sigma2_h in sigma2_h_list) {
    plot_single_loss(model_hist_list[[as.character(sigma2_h)]], sigma2_h)
  }
  
  # Global legend (last panel)
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend("center",
         legend = c("Training", "Validation"),
         col = adjustcolor(c("#1f77b4", "#ff7f0e"), alpha.f = 0.7),
         lty = c(1, 2), pch = c(16, 17), lwd = 1.5,
         horiz = TRUE, bty = "n", cex = 1.2)
  
  # Global title at the top (in outer margin)
  mtext(global_title,
        outer = TRUE, cex = 1, font = 2, line = 1)
  
  par(mfrow = c(1, 1))
}

plot_all_losses(nn_encoder_history_list, sigma2_h_list, 
                "Training and Validation Losses of Neural Networks \nfor the Fixed-effects Part of Mixed-effects VAE Models")

# ===============================
# Extracting fixed-effects estimates from the trained keras models in mixed-effects VAE
# ===============================
predict_image_batches <- function(nn_model, x_paths, 
                                  batch_size = 1000, target_size = c(64, 64)) {
  
  # Description:
  #   Predicts the fixed-effects estimates of the images from a trained keras model in mixed-effects VAE
  
  # Arguments:
  #   nn_model: a trained keras neural network model 
  #   x_paths: the file paths of the images to be used as inputs
  #   batch_size: the number of images to be included in each batch for the prediction of fixed-effects estimates
  #   target_size: the height and width of each image
  
  # Return:
  #   A matrix of the fixed-effects estimates in mixed-effects VAE for the input images
  
  n <- length(x_paths)
  preds <- matrix(NA, nrow = n, ncol = 2)
  
  for (i in seq(1, n, by = batch_size)) {
    upper <- min(i + batch_size - 1, n)
    batch_paths <- x_paths[i:upper]
    
    batch_x <- array(0, dim = c(length(batch_paths), target_size[1], target_size[2], 3))
    for (j in seq_along(batch_paths)) {
      img <- image_load(batch_paths[j], target_size = target_size)
      img_array <- image_to_array(img)
      img_array <- img_array / 255
      batch_x[j,,,] <- img_array
    }
    
    preds[i:upper, ] <- nn_model %>% predict(batch_x)
  }
  preds
}

# ===============================
# Random effects estimation from NN output
# ===============================

# Reformatting dataframe for brm
create_long_df = function(df, fixeff_preds, latent_dim) {
  
  # Description:
  #   Creates a dataframe in long format with the fixed-effects estimates of the latent variables for the images. 
  #   To be used for learning the random-effects estimates.
  
  # Arguments:
  #   df: dataframe containing information about the images
  #   fixeff_preds: fixed-effects estimates in mixed-effects VAE model for the images in `df`
  #   latent_dim: the dimension of each latent variable vector
  
  # Return:
  #   long_df: dataframe of the images with both their information and fixed-effects estimates in MVAE in long format
  
  
  long_df = data.frame(
    filename = rep(df$filename, times = 2),
    subject = rep(df$subject, times = 2),
    state = rep(df$state, times = 2),
    latent_dim = rep(paste0("h", seq_len(latent_dim)), 
                     each = nrow(df)), 
    latent_pred = c(fixeff_preds[, 1], fixeff_preds[, 2])
  )
  
  # Correct datatype
  long_df$subject = factor(long_df$subject)
  long_df$state = factor(long_df$state)
  long_df$latent_dim = factor(long_df$latent_dim)
  
  # Revelling response
  long_df$state = relevel(long_df$state, ref = "sleepy")
  
  # # Setup for brm fit
  # long_df$subject_dim = factor(interaction(long_df$subject, long_df$latent_dim))
  
  return(long_df)
}

# Estimating random effects for different sigma2_hs
for (sigma2_h in sigma2_h_list) {
  
  # No need for predicting random effects if sigma2_h = 0, i.e. independent latent variables
  brm_file_name = paste0("mvae_raneff/brm_raneff_sigma2h_", sigma2_h, ".rds")
  if (sigma2_h == 0 || file.exists(brm_file_name)) {
    next
  }
  
  # Loading fixed effects NN
  mvae_nn_name = paste0("mvae_fixeff/nn_with_sigma2h_", sigma2_h, ".keras")
  mvae_nn_model = load_model(mvae_nn_name)
  
  # Getting fixed-effects offset
  fixed_pred = predict_image_batches(mvae_nn_model, mrl_train_downsized$filename)
  
  long_train_df = create_long_df(mrl_train_downsized, fixed_pred, latent_dim)
  
  subjects = levels(long_train_df$subject)
  A = diag(1, length(subjects))  # 2D diagonal covariance structure
  dimnames(A) = list(subjects, subjects)
  
  # Fitting random effects model
  seed_val = sigma2_h*1e3
  
  brm_fit <- brm(
    formula = state ~ 0 + offset(latent_pred) + (1 | gr(subject, cov = A)),
    data = long_train_df,
    family = bernoulli(),
    data2 = list(A = A),
    # prior = set_prior(paste0("constant(", prior_sd, ")"), class = "sd"),     # constrain each latent dim variance to sigma2_h
    control = list(adapt_delta = 0.99),
    seed = seed_val
  )
  
  # Saving fitted brm object
  saveRDS(brm_fit, file = brm_file_name)
}

# ===============================
# Generating latent variable estimates from trained mixed VAE models
# ===============================
get_latent_pred = function(pred_data, latent_dim, fixeff_model, raneff_model = NULL) {
  
  # Description:
  #   Generates the encoding of an image from MRL eye dataset with a trained mixed-effects VAE model
  
  # Arguments:
  #   pred_data: dataframe of the images to be encoded
  #   latent_dim: the dimension of the latent variables
  #   fixeff_model: a trained keras model for the fixed-effects part
  #   raneff_model: random-effects estimates of the clusters seen in the training data. 
  #                 Defaults to NULL, which sets the random-effects estimates to 0.
  
  # Return:
  #   A dataframe containing the encoding of the images from the input dataframe.
  
  # Getting fixed-effects estimates
  fixeff_preds = predict_image_batches(fixeff_model, pred_data$filename)
  
  long_pred_data = create_long_df(pred_data, fixeff_preds, latent_dim)
  
  if (is.null(raneff_model)) {
    return(long_pred_data)
  }
  
  # And random effects
  raneff_preds = data.frame(ranef(raneff_model)$subject)
  known_subject_dims = rownames(raneff_preds)
  
  # Final latent variable prediction
  final_pred = long_pred_data$latent_pred + replace_na(raneff_preds[long_pred_data$subject, 1], 0)
  long_pred_data$latent_pred = final_pred
  
  return(long_pred_data)
}

# ===============================
# Training classifier neural network for evaluation
# ===============================

# Creating full filepaths for all the datasets
mrl_train_df_subset = create_full_filepath(mrl_train_df_subset, "train") # Only known 30 subjects
mrl_val_df_subset = create_full_filepath(mrl_val_df_subset, "val")# Only known 30 subjects
mrl_test_df = create_full_filepath(mrl_test_df, "test")

# Extracting test data accuracy for later
test_accu_list = list()
test_accu_list[["Baseline"]] = mean(mrl_test_df$state == "awake")  # 0.506

# Accuracy by whether subject is seen in training data
known_sub_idx = which(mrl_test_df$subject %in% cluster_split$known)

test_sub_known_accu_list = list()
test_sub_known_accu_list[["Baseline"]] = mean(mrl_test_df$state[known_sub_idx] == "awake")  # 0.54

test_sub_unknown_accu_list = list()
test_sub_unknown_accu_list[["Baseline"]] = mean(mrl_test_df$state[- known_sub_idx] == "awake")  # 0.402

# Classifier NN data pre-processing function 
nn_latent_func = function(df, latent_dim, fixeff_model, raneff_model = NULL) {
  
  # Description:
  #   Converts a dataframe in long format with encodings from mixed-effects VAE model 
  #   to a wide format suitable for training a classification neural network with keras
  
  # Arguments:
  #   df: dataframe of the images to be encoded
  #   latent_dim: the dimension of each latent variable vector
  #   fixeff_model: a trained keras model for the fixed-effects part
  #   raneff_model: random-effects estimates of the clusters seen in the training data. 
  #                 Defaults to NULL, which sets the random-effects estimates to 0.
  
  # Return:
  #   A wide format dataframe to be used for training a classification neural network in keras
  
  # Getting latent variable predictions on training data
  latent_pred_long = get_latent_pred(df, latent_dim,
                                     fixeff_model, raneff_model)
  
  # Converting latent variable encoding into wide format for keras
  latent_pred_wide = latent_pred_long %>% 
    pivot_wider(id_cols = c(filename, subject, state),
                names_from = latent_dim, 
                values_from = latent_pred)
  
  latent_pred_wide$state = as.numeric(latent_pred_wide$state) - 1
  
  return(latent_pred_wide)
}

# Classifier NN history
classifier_nn_hist_list = list()

for (sigma2_h in sigma2_h_list) {
  
  classifier_name = paste0("mrleye_classification/nn_classifier_sigma2h_", sigma2_h, ".keras")
  if (file.exists(classifier_name)) {
    next
  }
  
  # Fixed NN model
  mvae_nn_name = paste0("mvae_fixeff/nn_with_sigma2h_", sigma2_h, ".keras")
  mvae_nn_model = load_model(mvae_nn_name)
  
  # Random effects model 
  brm_file_name = paste0("mvae_raneff/brm_raneff_sigma2h_", sigma2_h, ".rds")
  
  if (file.exists(brm_file_name)) {
    brm_raneff_fit = readRDS(brm_file_name)
  } else {
    brm_raneff_fit = NULL
  } 
  
  # Getting latent variable predictions on training data
  train_latent_pred = nn_latent_func(mrl_train_df_subset, latent_dim,
                                     mvae_nn_model, brm_raneff_fit)
  
  # Validation data preparation
  val_latent_pred = nn_latent_func(mrl_val_df_subset, latent_dim, 
                                   mvae_nn_model, brm_raneff_fit)
  
  # Building binary classifier NN with latent variables as inputs
  
  hidden_layer = layer_dense(units = 100, activation = "relu")
  hidden_layer$build(shape(c(NULL, 2)))
  
  hidden_layer2 = layer_dense(units = 20, activation = "relu")
  hidden_layer2$build(shape(c(NULL, 100)))
  
  binary_dropout = layer_dropout(rate = 0.3)
  
  state_output = layer_dense(units = 1, activation = "sigmoid")
  state_output$build(shape(c(NULL, 20)))
  
  classifier_nn_model = keras_model_sequential(input_shape = 2) %>% 
    hidden_layer %>%
    hidden_layer2 %>%
    binary_dropout %>%
    state_output
  
  classifier_nn_model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy",
    metrics = c('accuracy')
  )
  
  # Early stopping 
  classifier_callbacks = list(callback_early_stopping(monitor = "val_accuracy",
                                                      patience = 5, 
                                                      restore_best_weights = TRUE, 
                                                      min_delta = 1e-3))
  
  # Fitting...
  classifier_hist = classifier_nn_model %>% fit(
    x = as.matrix(train_latent_pred[, c("h1", "h2")]),
    y = train_latent_pred$state, 
    epochs = 100, 
    batch_size = batch_size, 
    validation_data = list(as.matrix(val_latent_pred[, c("h1", "h2")]), 
                           val_latent_pred$state),
    callbacks = classifier_callbacks
  )
  
  classifier_nn_hist_list[[as.character(sigma2_h)]] = classifier_hist
  
  # Evaluation with test data
  test_latent_pred = nn_latent_func(mrl_test_df, latent_dim,
                                    mvae_nn_model, brm_raneff_fit)
  
  # Prediction on test data
  test_h_vecs = as.matrix(test_latent_pred[, c("h1", "h2")])
  
  test_state_prob_pred = classifier_nn_model %>% 
    predict(
      x = test_h_vecs
    )
  
  test_state_pred = as.vector(round(test_state_prob_pred))
  
  # Accuracy
  test_accu_list[[as.character(sigma2_h)]] = mean(test_latent_pred$state == test_state_pred)
  test_sub_known_accu_list[[as.character(sigma2_h)]] = mean(test_latent_pred$state[known_sub_idx] == test_state_pred[known_sub_idx])
  test_sub_unknown_accu_list[[as.character(sigma2_h)]] = mean(test_latent_pred$state[-known_sub_idx] == test_state_pred[-known_sub_idx])
  
  # Saving model
  save_model(classifier_nn_model, classifier_name)
}

# saveRDS(classifier_nn_hist_list, "nn_classifier_training_hist_v2.rds")
classifier_nn_hist_list = readRDS("mrleye_classification/nn_classifier_training_hist_v2.rds")

# Results of classifier model training
plot_all_losses(classifier_nn_hist_list, sigma2_h_list, 
               "Training and Validation Losses of Neural Network Classification Models for Drivers' States")

# Function to plot each subplot
plot_single_accuracy <- function(model_hist, sigma2_h, title = NULL) {
  
  # Description:
  #   Plots the evolution of the training and validation accuracy of a keras classification model 
  #   as a function of training epochs
  
  # Arguments:
  #   model_hist: the training history of a keras classification model with accuracy as the metric 
  #   sigma2_h: the element-wise covariance of the mixed-effects VAE model
  #   title: title of the plot, defaults to the value of sigma2_h used 
  
  # Return:
  #   A base R plot object
  
  n_epochs <- length(model_hist$metrics$loss)
  epochs = seq_len(n_epochs)
  accuracy <- model_hist$metrics$accuracy
  val_accuracy <- model_hist$metrics$val_accuracy
  y_range <- range(c(accuracy, val_accuracy))
  
  # Default title
  if (is.null(title)) {
    title = bquote(sigma[h]^2 == .(sigma2_h))
  }
  
  plot(epochs, accuracy, type = "n",
       ylim = y_range,
       xlab = "Epoch", ylab = "Accuracy",
       main = title,
       cex.main = 1.4, cex.axis = 1.1, 
       cex.lab = 1.2, axes = FALSE)
  
  # Custom integer x-axis
  x_ticks <- pretty(epochs, n = n_epochs)
  x_ticks <- x_ticks[x_ticks %% 1 == 0]
  axis(1, at = x_ticks, labels = x_ticks)
  axis(2)
  box()
  
  # Add accuracy and val_accuracy lines with transparency and markers
  lines(epochs, accuracy, type = "o",
        col = adjustcolor("#1f77b4", alpha.f = 0.7),
        lwd = 1.5, pch = 16)
  
  lines(epochs, val_accuracy, type = "o",
        col = adjustcolor("#ff7f0e", alpha.f = 0.7),
        lwd = 1.5, pch = 17, lty = 2)
}

# Function to plot 2x2 subplots of 4 sigma2_h values at once

plot_all_accuracies = function(model_hist_list, sigma2_h_list, global_title) {
  
  # Description:
  #   Plots a list of training histories of keras models 
  
  # Arguments:
  #   model_hist_list: a list of keras classification models training histories with accuracy as a metric
  #   sigma2_h_list: a list of values of sigma2_h corresponding to the list of keras models training histories
  #   global_title: the global title of all the subplots of keras model histories
  
  # Return:
  #   A 2x2 grid of subplots plotting all the model training histories 
  #   of the 4 values of sigma2_h considered in the manuscript
  
  # Reserve outer margin space for the global title (top = 4 lines)
  par(oma = c(0, 0, 4, 0))  # bottom, left, top, right
  
  # Create layout: title (mtext), 2x2 panels, bottom legend
  layout_matrix <- matrix(c(
    1, 2,
    3, 4,
    5, 5
  ), nrow = 3, byrow = TRUE)
  
  layout(layout_matrix, heights = c(1, 1, 0.25))
  
  # Draw the 4 subplots
  par(mar = c(4, 4, 3, 1))  # standard subplot margins
  for (sigma2_h in sigma2_h_list) {
    plot_single_accuracy(model_hist_list[[as.character(sigma2_h)]], sigma2_h)
  }
  
  # Global legend (last panel)
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend("center",
         legend = c("Training", "Validation"),
         col = adjustcolor(c("#1f77b4", "#ff7f0e"), alpha.f = 0.7),
         lty = c(1, 2), pch = c(16, 17), lwd = 1.5,
         horiz = TRUE, bty = "n", cex = 1.2)
  
  # Global title at the top (in outer margin)
  mtext(global_title,
        outer = TRUE, cex = 1, font = 2, line = 1)
  
  par(mfrow = c(1, 1))
}

# Plotting training and validation accuracies of all classification models
plot_all_accuracies(classifier_nn_hist_list, sigma2_h_list, 
                    "Training and Validation Accuracy Scores of Neural Network Classification Models for Drivers' States")

# Accuracy on test data into a LaTeX tablepaper
test_accu_df = data.frame(bind_rows(test_accu_list, # sigma2_h = 0.3 the best
                                    test_sub_known_accu_list, # sigma2_h = 0.3 the best
                                    test_sub_unknown_accu_list)) # sigma2_h = 0 the best
rownames(test_accu_df) = c("Overall", "Known subjects", "Unknown subjects")
saveRDS(test_accu_df, "classification_accuracies.rds")

kable(test_accu_df, "latex", digits = 4, 
      caption = "Accuracy score of feed-forward neural network in classifying driver's state, 
      with the inputs being latent variables encoded by mixed VAE models with different values of sigma^2_h", 
      label = "binary_test_accuracy")

# readRDS("nn_classifier_training_hist.rds")

# Getting random-effects estimates of MVAE model with sigma2_h = 0.3
best_brm_fit = readRDS("mvae_raneff/brm_raneff_sigma2h_0.3.rds")
brm_raneff_df = data.frame(ranef(best_brm_fit)$subject[, 1:2, 1])
kable(brm_raneff_df, "latex", digits = 4, 
      caption = "Random effects estimates of the mixed-effects VAE model with $\\sigma^2_h = 0.3$", 
      label = "tab:best_classifier_raneff")
