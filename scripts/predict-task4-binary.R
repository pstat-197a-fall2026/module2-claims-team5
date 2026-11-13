library(here)
library(glmnet)

# Load preprocessing functions

source(
  here(
    "scripts",
    "task4-binary-functions.R"
  )
)

# Load the unlabeled test data

load(
  here(
    "data",
    "claims-test.RData"
  )
)

# Load the fitted binary model

binary_model_bundle <- readRDS(
  here(
    "results",
    "task4-binary-model.rds"
  )
)

# Check the test data

stopifnot(
  nrow(claims_test) == 929
)

stopifnot(
  !anyDuplicated(claims_test$.id)
)

stopifnot(
  !anyNA(claims_test$text_tmp)
)

# Extract and clean test text

test_text <- preprocess_pages(
  claims_test
)

# Tokenize test text

test_counts <- tokenize_pages(
  test_text
)

# Create the test TF IDF matrix

x_test <- build_tfidf_matrix(
  counts = test_counts,
  page_ids = test_text$.id,
  vocabulary = binary_model_bundle$vocabulary
)

# Check the test matrix

stopifnot(
  nrow(x_test) == 929
)

stopifnot(
  identical(
    rownames(x_test),
    test_text$.id
  )
)

stopifnot(
  identical(
    colnames(x_test),
    binary_model_bundle$vocabulary$term
  )
)

# Predict binary probabilities

test_probabilities <- predict(
  binary_model_bundle$fit,
  newx = x_test,
  s = binary_model_bundle$lambda,
  type = "response"
) |>
  as.numeric()

# Check predicted probabilities

stopifnot(
  length(test_probabilities) == 929
)

stopifnot(
  !anyNA(test_probabilities)
)

stopifnot(
  all(
    test_probabilities >= 0 &
      test_probabilities <= 1
  )
)

# Convert probabilities to class labels

negative_class <- binary_model_bundle$class_levels[1]

positive_class <- binary_model_bundle$class_levels[2]

test_predictions <- ifelse(
  test_probabilities >=
    binary_model_bundle$threshold,
  positive_class,
  negative_class
)

test_predictions <- factor(
  test_predictions,
  levels = binary_model_bundle$class_levels
)

# Create the binary prediction data frame

binary_pred_df <- tibble::tibble(
  .id = test_text$.id,
  bclass.pred = test_predictions
)

# Check the final binary predictions

stopifnot(
  nrow(binary_pred_df) == 929
)

stopifnot(
  identical(
    binary_pred_df$.id,
    claims_test$.id
  )
)

stopifnot(
  !anyDuplicated(binary_pred_df$.id)
)

stopifnot(
  !anyNA(binary_pred_df$bclass.pred)
)

# Save the binary predictions

save(
  binary_pred_df,
  file = here(
    "results",
    "task4-binary-predictions.RData"
  )
)

# Print a prediction summary

print(
  table(
    binary_pred_df$bclass.pred
  )
)

print(
  binary_pred_df |>
    dplyr::slice_head(n = 10)
)