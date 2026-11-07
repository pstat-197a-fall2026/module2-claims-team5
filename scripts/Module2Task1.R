## TASK 1
# Note: I kept most of the professor's original NLP code but separated the TF-IDF steps so they happen after splitting the data. I also added code to make sure the 
# training and testing data use the same IDF values and word columns.

library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)

# load preprocessing functions
source('scripts/preprocessing.R')

url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
source(paste(url, 'projection-functions.R', sep = ''))

# load raw data
load('data/claims-raw.RData')


## PREPROCESSING

# original paragraphs only
claims_paragraphs <- claims_raw %>%
  parse_data()

# change to get headers too
parse_fn_headers <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n',
                            '[[:punct:]]',
                            'nbsp',
                            '[[:digit:]]',
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# function to apply to claims data adding headers
parse_data_headers <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_headers(text_tmp)) %>%
    unnest(text_clean)
  return(out)
}

# paragraphs and headers
claims_headers <- claims_raw %>%
  parse_data_headers()


## PARTITION 

# keep the same webpages for both models
common_ids <- intersect(claims_paragraphs$.id,
                        claims_headers$.id)

claims_paragraphs <- claims_paragraphs %>%
  filter(.id %in% common_ids)

claims_headers <- claims_headers %>%
  filter(.id %in% common_ids)

# check how many webpages are shared
length(common_ids)

# partition
set.seed(102926)
partitions <- claims_paragraphs %>%
  select(.id, bclass) %>%
  initial_split(prop = 0.8,
                strata = bclass
                )


## NLP

# use professor's nlp_fn steps before TF-IDF (correction) 
# TF-IDF calculated after partitioning
nlp_tokens <- function(.df){
  out <- .df %>%
    unnest_tokens(output = token,
                  input = text_clean,
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word,
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n')
  return(out)
}


## LOGISTIC PCR

# logistic regression on both datasets
fit_model <- function(.df){
  
  # use the same training and testing webpages for both models
  train <- .df %>%
    semi_join(training(partitions), by = '.id')
  
  test <- .df %>%
    semi_join(testing(partitions), by = '.id')
  
  # separate labels
  train_labels <- train %>%
    select(.id, bclass)
  
  test_labels <- test %>%
    select(.id, bclass)
  
  # apply professor's code
  train_tokens <- train %>%
    nlp_tokens()
  
  test_tokens <- test %>%
    nlp_tokens()
  
  # calculate TF-IDF with only training webpages
  train_tfidf <- train_tokens %>%
    bind_tf_idf(term = token.lem,
                document = .id,
                n = n)
  
  # save IDF values
  train_idf <- train_tfidf %>%
    distinct(token.lem, idf)
  
  # apply training IDF values to testing 
  test_tfidf <- test_tokens %>%
    group_by(.id) %>%
    mutate(tf = n / sum(n)) %>%
    ungroup() %>%
    inner_join(train_idf, by = 'token.lem') %>%
    mutate(tf_idf = tf * idf)
  
  # create training DTM
  train_dtm <- train_tfidf %>%
    select(.id, token.lem, tf_idf) %>%
    pivot_wider(id_cols = .id,
                names_from = token.lem,
                values_from = tf_idf,
                values_fill = 0)
  
  # create testing DTM
  test_dtm <- test_tfidf %>%
    select(.id, token.lem, tf_idf) %>%
    pivot_wider(id_cols = .id,
                names_from = token.lem,
                values_from = tf_idf,
                values_fill = 0)
  
  # keep webpages with no tokens
  train_dtm <- train_labels %>%
    select(.id) %>%
    left_join(train_dtm, by = '.id')
  
  test_dtm <- test_labels %>%
    select(.id) %>%
    left_join(test_dtm, by = '.id')
  
  # use the same word columns for training and testing
  vocabulary <- names(train_dtm)[-1]
  missing_words <- setdiff(vocabulary, names(test_dtm))
  
  test_dtm[missing_words] <- 0
  
  # remove IDs and replace missing TF-IDF values with zero
  train_dtm <- train_dtm %>%
    select(-.id) %>%
    mutate(across(everything(), ~replace_na(.x, 0)))
  
  test_dtm <- test_dtm %>%
    select(all_of(vocabulary)) %>%
    mutate(across(everything(), ~replace_na(.x, 0)))
  
  # find projections based on training data
  # changed to 0.8 to capture at least 80% of variance
  proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.8)
  
  train_dtm_projected <- proj_out$data
  
  # bind labels to projected data
  train <- train_labels %>%
    transmute(bclass = factor(bclass)) %>%
    bind_cols(train_dtm_projected)
  
  # store predictors and response as matrix and vector
  x_train <- train %>% select(-bclass) %>% as.matrix()
  y_train <- train_labels %>% pull(bclass)
  
  # fit enet model
  alpha_enet <- 0.3
  fit_reg <- glmnet(x = x_train,
                    y = y_train,
                    family = 'binomial',
                    alpha = alpha_enet)
  
  # choose a lambda by cross-validation
  set.seed(102926)
  cvout <- cv.glmnet(x = x_train,
                     y = y_train,
                     family = 'binomial',
                     alpha = alpha_enet)
  
  # store optimal lambda (penalty)
  lambda_opt <- cvout$lambda.min
  
  # project test data onto PCs
  test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)
  
  # coerce to matrix
  x_test <- as.matrix(test_dtm_projected)
  
  # compute predicted probabilities
  preds <- predict(fit_reg,
                   s = lambda_opt,
                   newx = x_test,
                   type = 'response')
  
  # store predictions in a data frame with true labels
  pred_df <- test_labels %>%
    transmute(bclass = factor(bclass,
                              levels = levels(factor(y_train)))) %>%
    bind_cols(pred = as.numeric(preds)) %>%
    mutate(bclass.pred = factor(pred > 0.5,
                                labels = levels(bclass)))
  
  # define classification metric panel
  panel <- metric_set(sensitivity,
                      specificity,
                      accuracy,
                      roc_auc)
  
  # compute test set accuracy
  results <- pred_df %>%
    panel(truth = bclass,
          estimate = bclass.pred,
          pred,
          event_level = 'second')
  
  # return accuracy so both models can be compared
  return(results %>%
           filter(.metric == 'accuracy') %>%
           pull(.estimate))
}


## COMPARE MODELS

# paragraphs only
accuracy_paragraphs <- fit_model(claims_paragraphs)

# paragraphs and headers
accuracy_headers <- fit_model(claims_headers)

# compare accuracy
results <- tibble(
  Model = c('Paragraphs only', 'Paragraphs and headers'),
  Accuracy = c(accuracy_paragraphs, accuracy_headers)
)

print(results)

# difference in accuracy by precent
accuracy_difference <- (accuracy_headers - accuracy_paragraphs) * 100

print(accuracy_difference)
