library("ggplot2")
library("tidyr")
source("data_read.R")

# some quick findings:
# * we need to tune one model per output element e.g. HandStart
# * Signals seem to vary among subjects => normalizations or per subject online learning or both
# * one option could also be to combine models to use the sequence structure of the events. E.g. a HandStart can only occur first in the series or if the BothReleased classifier identified that event. 

##Load the training data
eegTrain <- get_datasets(subjects=1:2, series = 1:3, verbose=TRUE)

# plot some sensors for a single series and individual
data_plot <- eegTrain %>% 
  filter(subject == 1 & series == 1) %>%
  #filter(series_seq_id %% 2 == 0) %>% # take every second data point
  filter(series_seq_id <= 4000) %>% #let us only consider one session for now
  select(-id) %>%
  gather(feature, value, Fp1:PO10) %>%
  gather(output, output_value, HandStart:BothReleased) %>%
  mutate(output_value = ifelse(output_value == 0, NA, 1)) 

ggplot(data = data_plot, aes(x = series_seq_id)) + 
  facet_wrap(~ feature) + 
  geom_line(aes(y = value)) + 
  geom_line(aes(y = output_value, color = output), size = 2)

# Nothing extremly special, but we need to check of outliers. 
# Also it shows that a couple of features spike when HandStart event
# Let us now explore the HandStart event over different individuals and series
subject_data <- get_datasets(series = 1, verbose=TRUE)

data_plot <- subject_data %>% 
  filter(HandStart == 1) %>%
  group_by(subject, series) %>%
  mutate(seq_id_rel = series_seq_id - min(series_seq_id)) %>%
  #filter(series_seq_id %% 2 == 0) %>% # take every second data point
  filter(series_seq_id <= 2000) %>% #let us only consider one session for now
  mutate(subj_series = factor(paste0(subject, series))) %>%
  select(-id) %>%
  gather(feature, value, Fp1:PO10) %>%
  group_by(subject, series, feature) %>%
  mutate(value_normal = (value - mean(value)) / sd(value)) 
ggplot(data = data_plot, aes(x = seq_id_rel)) + 
  facet_wrap(~ feature) + 
  geom_line(aes(y = value, color = subject, group = factor(subj_series))) 

# it looks like signals are different per subject, but when normalized it again looks alright