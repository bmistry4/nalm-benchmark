rm(list = ls())
# ofile= 'C:\\Users\\Bhumika\\Documents\\SOTON\\PhD\\Code\\stable-nalu\\export\\function_task_static.r'
# setwd(dirname(parent.frame(2)$ofile))

# minerva example ->   Rscript fts.r ../../../../../../datasets/bm4g15/nalu-stable-exp/csvs/ ../../../../../../datasets/bm4g15/nalu-stable-exp/csvs/r_results/ nau-nmu-mul_E5M
args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results 
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
csv_ext = '.csv'

library(plyr)
library(dplyr)
library(tidyr)
library(readr)
source('./_function_task_expand_name.r')
source('./_function_task_table.r')
source('./_compute_summary.r')

best.range = 5000

best.model.step.fn = function (errors) {
  best.step = max(length(errors) - best.range, 0) + which.min(tail(errors, best.range))
  if (length(best.step) == 0) {
    return(length(errors))
  } else {
    return(best.step)
  }
}

first.solved.step = function (steps, errors, threshold) {
  index = first(which(errors < threshold))
  if (is.na(index)) {
    return(NA)
  } else {
    return(steps[index])
  }
}

# eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
eps = read_csv('../function_task_static_mse_expectation.csv') %>%
  filter(simple == FALSE & parameter == 'default') %>%
  mutate(
    operation = revalue(operation, operation.full.to.short),
  ) %>%
  select(operation, threshold)

dat = expand.name(
  # read_csv('../results/function_task_static.csv', col_types=cols(sparse.error.max=col_double()))
  read_csv(paste(load_folder, base_filename, csv_ext, sep=''), col_types=cols(sparse.error.max=col_double()))
) %>%
  merge(eps)

dat.last = dat %>%
  group_by(name) %>%
  #filter(n() == 201) %>%
  summarise(
    threshold = last(threshold),
    best.model.step = best.model.step.fn(metric.valid.interpolation),
    interpolation.last = metric.valid.interpolation[best.model.step],
    extrapolation.last = metric.test.extrapolation[best.model.step],
    interpolation.step.solved = first.solved.step(step, metric.valid.interpolation, threshold),
    extrapolation.step.solved = first.solved.step(step, metric.test.extrapolation, threshold),
    sparse.error.max = sparse.error.max[best.model.step],
    solved = replace_na(metric.test.extrapolation[best.model.step] < threshold, FALSE),
    model = last(model),
    operation = last(operation),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation) %>%
  group_modify(compute.summary) %>%
  ungroup()

# print(dat.last.rate)
print.data.frame(dat.last.rate)

write.csv(dat.last, paste(results_folder, base_filename, '_seeds_best', csv_ext, sep=''))            # best result for each seed
write.csv(dat.last.rate, paste(results_folder, base_filename, '_final_metrics', csv_ext, sep='')) # Madsen eval metrics with confidence intervals 
print("R Script completed.")