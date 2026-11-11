#Module2Task2

# Module 2 Task 2: Compare unigrams with unigrams plus bigrams.
# Run after Module2Task1.R in the same R session.

# 1. using task 1's cleaned text and train/test split
pages <- claims_headers %>% ungroup()

train_pages <- pages %>%
  semi_join(training(partitions), by = ".id")

test_pages <- pages %>%
  semi_join(testing(partitions), by = ".id")

y_train <- factor(train_pages$bclass)
class_levels <- levels(y_train)

# 2. convert term counts into matching train/test TF-IDF matrices
make_tfidf_matrices <- function(counts, max_terms = Inf, min_pages = 1) {
  # calculate each term's frequency within its page
  counts <- counts %>%
    group_by(.id) %>%
    mutate(tf = n / sum(n)) %>%
    ungroup()
  
  # learn the vocabulary and IDF weights from training pages
  train_counts <- counts %>%
    semi_join(train_pages, by = ".id")
  
  vocabulary <- train_counts %>%
    dplyr::count(term, name = "n_pages", sort = TRUE) %>%
    filter(n_pages >= min_pages) %>%
    mutate(idf = log(n_distinct(train_counts$.id) / n_pages))
  
  if (is.finite(max_terms)) {
    vocabulary <- vocabulary %>% slice_head(n = max_terms)
  }
  
  # keep every page in order and use the same term columns for both splits
  build_matrix <- function(page_ids) {
    weighted <- counts %>%
      filter(.id %in% page_ids) %>%
      inner_join(vocabulary, by = "term")
    
    Matrix::sparseMatrix(
      i = match(weighted$.id, page_ids),
      j = match(weighted$term, vocabulary$term),
      x = weighted$tf * weighted$idf,
      dims = c(length(page_ids), nrow(vocabulary)),
      dimnames = list(NULL, vocabulary$term)
    )
  }
  
  list(
    train = build_matrix(train_pages$.id),
    test = build_matrix(test_pages$.id)
  )
}

# 3. fit the unigram model and get one log-odds score per page
unigram_counts <- pages %>%
  nlp_tokens() %>%
  select(.id, term = token.lem, n)

unigram_dtm <- make_tfidf_matrices(unigram_counts)

# learn word PCs on training pages and apply them to test pages
unigram_projection <- projection_fn(
  .dtm = as_tibble(as.matrix(unigram_dtm$train)),
  .prop = 0.8
)

unigram_x_train <- as.matrix(unigram_projection$data)
unigram_x_test <- as.matrix(reproject_fn(
  .dtm = as_tibble(as.matrix(unigram_dtm$test)),
  unigram_projection
))

set.seed(102926)
unigram_fit <- glmnet::cv.glmnet(
  x = unigram_x_train,
  y = y_train,
  family = "binomial",
  alpha = 0.3,
  keep = TRUE
)

# use held-out logistic fold scores for the training pages
lambda_col <- which.min(abs(unigram_fit$lambda - unigram_fit$lambda.min))
train_log_odds <- as.numeric(unigram_fit$fit.preval[, lambda_col])

# score test pages with the fit on the full training split
test_log_odds <- as.numeric(predict(
  unigram_fit,
  newx = unigram_x_test,
  s = "lambda.min",
  type = "link"
))

# 4. count adjacent word pairs and reduce them to 20 bigram PCs
bigram_counts <- pages %>%
  select(.id, text_clean) %>%
  unnest_tokens(
    output = term,
    input = text_clean,
    token = "ngrams",
    n = 2
  ) %>%
  filter(!is.na(term)) %>%
  dplyr::count(.id, term, name = "n")

# keep up to 1,000 phrases found on at least five training pages
bigram_dtm <- make_tfidf_matrices(
  bigram_counts,
  max_terms = 1000,
  min_pages = 5
)

bigram_pca <- prcomp(as.matrix(bigram_dtm$train), rank. = 20)

bigram_train_pcs <- as_tibble(bigram_pca$x)
bigram_test_pcs <- as_tibble(predict(
  bigram_pca,
  newdata = as.matrix(bigram_dtm$test)
))

# 5. fit a second logistic regression using the word score and phrase PCs
combined_train <- bind_cols(
  tibble(bclass = y_train, unigram_log_odds = train_log_odds),
  bigram_train_pcs
)

combined_fit <- glm(
  bclass ~ .,
  data = combined_train,
  family = binomial()
)

# 6. compare both models on the same test pages
combined_test <- bind_cols(
  tibble(unigram_log_odds = test_log_odds),
  bigram_test_pcs
)

unigram_prob <- plogis(test_log_odds)
combined_prob <- predict(
  combined_fit,
  newdata = combined_test,
  type = "response"
)

# apply the same 0.5 cutoff to both models
task2_predictions <- tibble(
  .id = test_pages$.id,
  actual = as.character(test_pages$bclass),
  unigram_pred = if_else(
    unigram_prob > 0.5, class_levels[2], class_levels[1]
  ),
  combined_pred = if_else(
    combined_prob > 0.5, class_levels[2], class_levels[1]
  )
)

# print both accuracies and the change in percentage points
task2_results <- task2_predictions %>%
  summarise(
    unigram_accuracy_pct = 100 * mean(unigram_pred == actual),
    combined_accuracy_pct = 100 * mean(combined_pred == actual),
    difference_pp = combined_accuracy_pct - unigram_accuracy_pct
  )

print(task2_results)


# On the held-out pages, unigram accuracy was 78.3%.
# Adding 20 bigram PCs lowered accuracy to 72.3%, a drop of 6.06 percentage points.
# The combined model also produced some training probabilities near 0 or 1.