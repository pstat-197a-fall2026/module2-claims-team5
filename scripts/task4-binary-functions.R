extract_page_text <- function(html) {
  tryCatch(
    {
      document <- rvest::read_html(html)
      
      page_text <- document |>
        rvest::html_elements(
          "title, h1, h2, h3, h4, h5, h6, p"
        ) |>
        rvest::html_text2() |>
        stringr::str_c(collapse = " ") |>
        stringr::str_squish()
      
      if (stringr::str_length(page_text) == 0) {
        page_text <- document |>
          rvest::html_text2() |>
          stringr::str_squish()
      }
      
      page_text
    },
    error = function(e) {
      ""
    }
  )
}

clean_page_text <- function(text) {
  text |>
    stringr::str_replace_all(
      "https?://\\S+|www\\.\\S+",
      " "
    ) |>
    stringr::str_replace_all(
      "\\b[[:alnum:]._%+-]+@[[:alnum:].-]+\\.[[:alpha:]]{2,}\\b",
      " "
    ) |>
    stringr::str_replace_all(
      "([a-z])([A-Z])",
      "\\1 \\2"
    ) |>
    stringr::str_to_lower() |>
    stringr::str_replace_all(
      "[^a-z0-9\\s]",
      " "
    ) |>
    stringr::str_squish()
}

preprocess_pages <- function(data) {
  data |>
    dplyr::mutate(
      text_clean = purrr::map_chr(
        text_tmp,
        function(html) {
          extracted_text <- extract_page_text(html)
          clean_page_text(extracted_text)
        }
      )
    )
}

tokenize_pages <- function(data) {
  data |>
    dplyr::select(
      .id,
      text_clean
    ) |>
    tidytext::unnest_tokens(
      output = term,
      input = text_clean,
      token = "words"
    ) |>
    dplyr::filter(
      stringr::str_detect(
        term,
        "^[a-z]+$"
      ),
      stringr::str_length(term) > 2
    ) |>
    dplyr::anti_join(
      tidytext::stop_words,
      by = c(
        "term" = "word"
      )
    ) |>
    dplyr::count(
      .id,
      term,
      name = "n"
    )
}

build_tfidf_matrix <- function(
    counts,
    page_ids,
    vocabulary
) {
  weighted_counts <- counts |>
    dplyr::group_by(.id) |>
    dplyr::mutate(
      tf = n / sum(n)
    ) |>
    dplyr::ungroup() |>
    dplyr::inner_join(
      vocabulary |>
        dplyr::select(
          term,
          idf
        ),
      by = "term"
    ) |>
    dplyr::mutate(
      tf_idf = tf * idf
    )
  
  row_index <- match(
    weighted_counts$.id,
    page_ids
  )
  
  column_index <- match(
    weighted_counts$term,
    vocabulary$term
  )
  
  stopifnot(
    !anyNA(row_index)
  )
  
  stopifnot(
    !anyNA(column_index)
  )
  
  stopifnot(
    all(
      page_ids[row_index] ==
        weighted_counts$.id
    )
  )
  
  stopifnot(
    all(
      vocabulary$term[column_index] ==
        weighted_counts$term
    )
  )
  
  Matrix::sparseMatrix(
    i = row_index,
    j = column_index,
    x = weighted_counts$tf_idf,
    dims = c(
      length(page_ids),
      nrow(vocabulary)
    ),
    dimnames = list(
      page_ids,
      vocabulary$term
    )
  )
}