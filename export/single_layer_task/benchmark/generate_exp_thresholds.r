rm(list = ls())
# setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)

seed = 1234
set.seed(seed)

# ( '[-40,-20]' '[-6,-2]' '[-6.1,-1.2]' '[-2,-0.2]'  '[[-6,-2],[2,6]]' '[0.2,2]'   '[2,6]' '[1.2,6]'  '[20,40]' )
simulate.mse = function(epsilon, samples, operation, simple, input.size, subset.ratio, overlap.ratio, range.a, range.b, range.mirror) {
  # matrix = (data, row, cols)
  # runif = generate random numbers of len of the 1st arg with ranges a to b
  X = matrix(runif(samples * input.size, range.a, range.b), samples, input.size)

  if (range.mirror) {
    # rbinom(n (|obvs|), size, prob) = rbinom(number of runs, ntrials, success prob)
    # rbinom(400, size = 10, prob = 0.2) = gives the results of 400 runs of 10 coin flips each, returning the number of successes in each run.
    # creates a sign matrix (of -1 and 1s) using a binomial dist
    X.sign = matrix(rbinom(samples * input.size, 1, 0.5), samples, input.size) * 2 - 1
    X = X * X.sign
  }

  evaluate = function(epsilon) {
    # X = [N, in], W = [In, 1], Y = [N,1]
    subset.size = floor(subset.ratio * input.size)
    a.start = 1
    a.end = a.start + subset.size

    # single layer operation
    if (operation == 'add') {
      Y = matrix(rowSums(X[, a.start:a.end]) - (rowSums(abs(X)) * epsilon))
    } else if (operation == 'sub') {
      Y = matrix((X[, a.start] - X[, a.end]) - (rowSums(abs(X)) * epsilon))
    } else if (operation == 'mul') {
      relevant.inp.calc = (1 - epsilon)^2 * apply(X[, a.start:a.end], 1, prod)
      # if there's no irrelevant inputs then we'll just multiply by 1
      if (a.end == ncol(X)) {
        irrelevant.inp.calc = 1
      }else {
        irrelevant.inp.calc = apply(1 - (epsilon * abs(X[, (a.end + 1):ncol(X)])), 1, prod)
      }
      Y = matrix(relevant.inp.calc * irrelevant.inp.calc)
    } else if (operation == 'div') {
      relevant.inp.calc = ((1 - epsilon) / (1 + epsilon)) * (X[, a.start] / X[, a.end])
      # if there's no irrelevant inputs then we'll just multiply by 1
      if (a.end == ncol(X)) {
        irrelevant.inp.calc = 1
      }else {
        irrelevant.inp.calc = apply(1 - (epsilon * abs(X[, (a.end + 1):ncol(X)])), 1, prod)
      }
      Y = matrix(relevant.inp.calc * irrelevant.inp.calc)
    }
    return(Y)
  }

  errors = (evaluate(epsilon) - evaluate(0))**2
  return(mean(errors))
}

# ( '[-40,-20]' '[-6,-2]' '[-6.1,-1.2]' '[-2,-0.2]'  '[[-6,-2],[2,6]]' '[0.2,2]'   '[2,6]' '[1.2,6]'  '[20,40]' )
cases = rbind()
for (operation in c('add', 'sub', 'mul', 'div')) {
  cases = rbind(cases,
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 2, subset.ratio = 0.5, overlap.ratio = 0, range.a = -40, range.b = -20, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 2, subset.ratio = 0.5, overlap.ratio = 0, range.a = -6, range.b = -2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 2, subset.ratio = 0.5, overlap.ratio = 0, range.a = -6.1, range.b = -1.2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 2, subset.ratio = 0.5, overlap.ratio = 0, range.a = -2, range.b = -0.2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 2, subset.ratio = 0.5, overlap.ratio = 0, range.a = 2, range.b = 6, range.mirror = T),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 2, subset.ratio = 0.5, overlap.ratio = 0, range.a = 0.2, range.b = 2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 2, subset.ratio = 0.5, overlap.ratio = 0, range.a = 2, range.b = 6, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 2, subset.ratio = 0.5, overlap.ratio = 0, range.a = 1.2, range.b = 6, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 2, subset.ratio = 0.5, overlap.ratio = 0, range.a = 20, range.b = 40, range.mirror = F)
  )
}
for (operation in c('add', 'sub')) {
  cases = rbind(cases,
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 10, subset.ratio = 0.1, overlap.ratio = 0, range.a = -40, range.b = -20, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 10, subset.ratio = 0.1, overlap.ratio = 0, range.a = -6, range.b = -2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 10, subset.ratio = 0.1, overlap.ratio = 0, range.a = -6.1, range.b = -1.2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 10, subset.ratio = 0.1, overlap.ratio = 0, range.a = -2, range.b = -0.2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 10, subset.ratio = 0.1, overlap.ratio = 0, range.a = 2, range.b = 6, range.mirror = T),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 10, subset.ratio = 0.1, overlap.ratio = 0, range.a = 0.2, range.b = 2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 10, subset.ratio = 0.1, overlap.ratio = 0, range.a = 2, range.b = 6, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 10, subset.ratio = 0.1, overlap.ratio = 0, range.a = 1.2, range.b = 6, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 10, subset.ratio = 0.1, overlap.ratio = 0, range.a = 20, range.b = 40, range.mirror = F),

                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 100, subset.ratio = 0.01, overlap.ratio = 0, range.a = -40, range.b = -20, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 100, subset.ratio = 0.01, overlap.ratio = 0, range.a = -6, range.b = -2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 100, subset.ratio = 0.01, overlap.ratio = 0, range.a = -6.1, range.b = -1.2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 100, subset.ratio = 0.01, overlap.ratio = 0, range.a = -2, range.b = -0.2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 100, subset.ratio = 0.01, overlap.ratio = 0, range.a = 2, range.b = 6, range.mirror = T),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 100, subset.ratio = 0.01, overlap.ratio = 0, range.a = 0.2, range.b = 2, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 100, subset.ratio = 0.01, overlap.ratio = 0, range.a = 2, range.b = 6, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 100, subset.ratio = 0.01, overlap.ratio = 0, range.a = 1.2, range.b = 6, range.mirror = F),
                c(parameter = 'extrapolation.range', operation = operation, simple = F, input.size = 100, subset.ratio = 0.01, overlap.ratio = 0, range.a = 20, range.b = 40, range.mirror = F)
  )
}

eps = data.frame(rbind(
  c(operation = 'mul', epsilon = 0.0001),
  c(operation = 'add', epsilon = 0.0001),
  c(operation = 'sub', epsilon = 0.0001),
  c(operation = 'div', epsilon = 0.0001)
))

dat = data.frame(cases) %>%
  merge(eps) %>%
  mutate(
    simple = as.logical(as.character(simple)),
    input.size = as.integer(as.character(input.size)),
    subset.ratio = as.numeric(as.character(subset.ratio)),
    overlap.ratio = as.numeric(as.character(overlap.ratio)),
    range.a = as.numeric(as.character(range.a)),
    range.b = as.numeric(as.character(range.b)),
    range.mirror = as.logical(as.character(range.mirror)),
    epsilon = as.numeric(as.character(epsilon))
  ) %>%
  rowwise() %>%
  mutate(
    threshold = simulate.mse(epsilon, 1000000, operation, simple, input.size, subset.ratio, overlap.ratio, range.a, range.b, range.mirror),
    extrapolation.range = ifelse(range.mirror, ifelse(range.b <= 0, paste0('U[', range.b * -1, ',', range.a * -1, '] & U[', range.a, ',', range.b, ']'), paste0('U[-', range.b, ',-', range.a, '] & U[', range.a, ',', range.b, ']')), paste0('U[', range.a, ',', range.b, ']')), # if a and b are -ve then want: [[+,+],[-,-]]
    operation = paste0('op-', operation)
  )

# TODO - set samples to 1000000
write.csv(dat, file = "1e-4_eps_thrs_sltr.csv", row.names = F)
print("Script completed.")
