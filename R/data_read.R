library("dplyr")
library("readr")
library("stringr")

# this file should contain functions to extract the data and generate training, 
# validation and test sets

# This function returns a single data frame with data and events combined for all subjects and series.
# example usage: 
# library("ggplot2")
# t <- get_datasets(1, 1:8) 
# ggplot(t, aes(x = series_seq_id, y = F7)) + facet_wrap(~ series) + geom_line()
get_datasets <- function(subjects = 1:12, series = 1:8, base_path = "../Data/train") {
  construct_filename <- function(subject, series_id, type) {
    paste0(base_path, "/subj", subject, "_series", series_id, "_", type, ".csv")
  }
  
  # could be done in parallel
  expand.grid(subject = subjects, series = series) %>%
    apply(1, function(row) {
      events <- read_csv(construct_filename(row[["subject"]], row[["series"]], "events"))
      data <- read_csv(construct_filename(row[["subject"]], row[["series"]], "data"))
      data %>% left_join(events, by = "id")
    }) %>%
    bind_rows %>%
    mutate(subject = str_match(id, "subj([[:digit:]]{1,2})")[, 2],
           series = str_match(id, "series([[:digit:]])")[, 2],
           series_seq_id = str_match(id, "_([[:digit:]]+)")[, 2]) %>%
    mutate(subject = as.numeric(subject), 
           series = as.numeric(series), 
           series_seq_id = as.numeric(series_seq_id))
}

