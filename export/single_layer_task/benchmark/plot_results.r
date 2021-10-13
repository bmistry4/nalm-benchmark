rm(list = ls())
#setwd(dirname(parent.frame(2)$ofile))

args <- commandArgs(trailingOnly = TRUE)
load_folder <- args[1]    # folder to load csv with converted tensorboard results 
results_folder <- args[2] # folder to save the generated r results
base_filename <- args[3]  # base name which influences loading and saving of files (e.g. function_task_static.csv)
op <- args[4] # operation filter on (e.g. op-mul)
model_name <- args[5] # name of model to use in plot (i.e. short name). Use 'None' if you don't want to change the default model name. To be used on results file with only one model.
model_name=ifelse(is.na(model_name), 'None', model_name)  # no passed arg becomes 'None' i.e. use default name
merge_mode <- args[6] # if 'None' then just loads single file. Otherwise looks up multiple results to merge together (use when have multiple models to plot)
merge_mode=ifelse(is.na(merge_mode), 'None', merge_mode)  # no passed arg becomes 'None' i.e. single model plot
parameter_value <- args[7]  # type of experiment e.g. extrapolation.ranges (see exp_setups for value options)
parameter_value=ifelse(is.na(parameter_value), 'extrapolation.range', parameter_value) # is no argument given then assume you want extrapolation.range
csv_ext = '.csv'

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(xtable)
source('../_single_layer_task_expand_name.r')
source('../../_compute_summary.r')
source('../../_plot_parameter.r')
source('./_table.r')
source('./csv_merger.r')

# number of logged steps to look at (starting from the last step and working backwards).
best.range = 5000

# find the step with the lowest error in the allowed range of steps.
# a range larger than the errors length will just consider all the rows in errors
best.model.step.fn = function (errors) {
  best.step = max(length(errors) - best.range, 0) + which.min(tail(errors, best.range))
  if (length(best.step) == 0) {
    return(length(errors))
  } else {
    return(best.step)
  }
}

# return the first step where the error is under the allowed threshold. If none exists, return NA.
first.solved.step = function (steps, errors, threshold) {
  index = first(which(errors < threshold))
  if (is.na(index)) {
    return(NA)
  } else {
    return(steps[index])
  }
}

name.parameter = 'interpolation.range'  # column name containing x-axis values
name.label = 'Interpolation range'      # x-axis label
name.file = paste0(load_folder, base_filename, csv_ext)
name.output = paste0(results_folder, base_filename, '_', op)

# load the experiment setups file containing the thresholds and filter for the relevant experiments.
eps = read_csv('./exp_setups.csv') %>%
  filter(simple == FALSE & parameter == parameter_value & operation == op) %>%
  mutate(
    operation = revalue(operation, operation.full.to.short)
  ) %>%
  select(operation, extrapolation.range, epsilon)

# load (and merge) the exp results csvs and merge with the experument setup files
dat = load.and.merge.csvs(merge_mode)  %>%
  # to maintain ordering of dat use join not merge (otherwise the solved at subplot will be incorrect)
  inner_join(eps)  %>%
  mutate(
    # !! = remember the expression I stored recently? Now take it, and ‘unquote’ it, that is, just run it!”
    parameter = !!as.name(name.parameter)
  )

dat.last = dat %>%
  group_by(name, parameter) %>%
  summarise(
    threshold = last(epsilon),
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
  group_by(model, operation, parameter) %>%
  group_modify(compute.summary)

if (model_name != 'None') {
  # Rename model with the given arg. The column is a factor, so the levels require renaming.
  levels(dat.last.rate$model) <- model_name
}

dat.last.rate$parameter <- gsub(']', ')', dat.last.rate$parameter)  # replace interp range notation from inclusion to exclusion i.e. ] to )

dat.gather = plot.parameter.make.data(dat.last.rate)

p = ggplot(dat.gather, aes(x = parameter, colour=model, group=interaction(parameter, model))) +
  geom_point(aes(y = mean.value), position=position_dodge(width=0.3)) +
  geom_errorbar(aes(ymin = lower.value, ymax = upper.value), position=position_dodge(width=0.3), alpha=0.5) +
  # for a custom label order replace labels = model.to.exp(levels(dat.gather$model)) with limits=c(<"model1">, <"model2">, <"model3">) (copied from csv_merger.r)
  #scale_color_discrete("", limits=c('Real NPU')) +         # legend title and ordering
  scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
  scale_x_discrete(name = name.label) +
  scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
  scale_shape(guide = FALSE) +
  facet_wrap(~ key, scales='free_y', labeller = labeller(
    key = c(
      success.rate = "Extrapolation range success rate",
      converged.at = "Solved at iteration step",
      sparse.error = "Sparsity error"
    )
  )) +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points")) +
  theme(panel.spacing.x=unit(1, "lines")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) #+
  #geom_point(aes(y = max.value), position=position_dodge(width=0.3), shape=18) #+ # plots max solved at step

# SAVE FILES
ggsave(paste0(name.output, '.pdf'), p, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")
write.csv(dat.gather, paste(results_folder, base_filename, '_', op, '_plot_data.csv', sep=''))  # save results table
#write.csv(dat.last, paste(results_folder, base_filename, '_seeds_best', csv_ext, sep=''))            # best result for each seed
#write_csv(filter(dat.last, solved == FALSE) %>% select(parameter, seed),  paste(results_folder, base_filename, '_seeds_failure', csv_ext , sep=''))

op_long = switch(
  op,
  "op-add" = "addition",
  "op-sub" = "subtraction",
  "op-mul" = "multiplication",
  "op-div" = "division",
)

save.table(
  dat.last.rate,
  paste0("benchmark-sltr-", op),
  paste0(paste0("Results for ", op_long), ". Comparison of the success-rate, model convergence iteration, and the sparsity error, with 95\\% confidence interval on the ``single layer'' task. Each value is a summary of 25 different seeds. \
  Bold values refers to the best result for a evaluation metric for a single module across the different ranges."),
  paste0(name.output, ".tex"),
  longtable=T,
  show.operation=F
)

print("R Script completed.")
