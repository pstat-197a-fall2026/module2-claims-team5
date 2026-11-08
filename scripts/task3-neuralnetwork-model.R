# ==========================================
# TASK 3: NEURAL NETWORK
# ==========================================

# STEP 1: LOAD PACKAGES AND DATA
library(tidyverse)
library(tidymodels)
library(tensorflow)
library(reticulate)
library(keras3)

# Use the working Python environment
use_python(
  path.expand("~/.virtualenvs/r-tensorflow-py312/bin/python"),
  required = TRUE
)

# Load cleaned claims data
load("data/claims-clean-example.RData")


# STEP 2: SPLIT THE DATA
set.seed(102928)

partitions <- initial_split(
  claims_clean,
  prop = 0.8,
  strata = bclass
)

train_data <- training(partitions)
valid_data <- testing(partitions)

# Extract text
train_text <- train_data %>% pull(text_clean)
valid_text <- valid_data %>% pull(text_clean)

# Convert binary labels to 0 and 1
train_labels <- as.numeric(train_data$bclass) - 1
valid_labels <- as.numeric(valid_data$bclass) - 1


# STEP 3: CREATE THE TEXT PREPROCESSING LAYER
set.seed(102928)
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = "whitespace",
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = "tf_idf"
)

# Learn vocabulary and IDF weights from training text only
adapt(preprocess_layer, train_text)



# ==========================================
# EXPERIMENT 1: 50 HIDDEN UNITS
# ==========================================

# STEP 4: BUILD THE MODEL
input <- layer_input(shape = c(1), dtype = "string")
set.seed(102928)

output <- input |>
  preprocess_layer() |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 50) |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 1) |>
  layer_activation(activation = "sigmoid")

model <- keras_model(
  inputs = input,
  outputs = output
)

summary(model)


# STEP 5: COMPILE
set.seed(102928)
compile(
  model,
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("binary_accuracy")
)


# STEP 6: TRAIN
set.seed(102928)
history <- fit(
  model,
  x = train_text,
  y = train_labels,
  epochs = 5,
  batch_size = 32,
  validation_data = list(valid_text, valid_labels)
)


# STEP 7: EVALUATE
set.seed(102928)
pred_probs <- predict(model, valid_text)

pred_labels <- ifelse(
  as.vector(pred_probs) >= 0.5, 1, 0
)

accuracy_50 <- mean(pred_labels == valid_labels)

accuracy_50

table(
  Actual = valid_labels,
  Predicted = pred_labels
)

# ==========================================
# STEP 7: SAVE THE FITTED MODEL
# ==========================================

# Export model in TensorFlow SavedModel format
export_savedmodel(
  model,
  export_dir_base = "results/task3-neural-network-model"
)