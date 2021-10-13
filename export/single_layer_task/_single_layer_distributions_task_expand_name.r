# use this when you want to parse different distributions e.g. uniform, benford, truncated normal.

library(latex2exp)

single.model.full.to.short = c(
  'nalu'='NALU',
  'reregualizedlinearnac'='NAU',
  'reregualizedlinearmnac'='NMU',
  'realnpu'='Real NPU',
  'npu'='NPU',
  'nac'='$\\mathrm{NAC}_{+}$',
  'mnac'='$\\mathrm{NAC}_{\\bullet}$'
)

model.full.to.short = c(
  'linear'='Linear',
  'relu'='ReLU',
  'relu6'='ReLU6',
  'nac'='$\\mathrm{NAC}_{+}$',
  'nac-nac-n'='$\\mathrm{NAC}_{\\bullet}$',
  'posnac-nac-n'='$\\mathrm{NAC}_{\\bullet,\\sigma}$',
  'nalu'='NALU',
  'nalu-nalu-2n'='NALU (separate)',
  'reregualizedlinearnac'='NAU',
  'reregualizedlinearnac-nac-m'='NMU',
  'reregualizedlinearnalu-nalu-2nm'='Gated NAU/NMU',
  'reregualizedlinearposnac-nac-n'='$\\mathrm{NAC}_{\\bullet,\\mathrm{NMU}}$',
  'regualizedlinearnac-nac-m'='NMU, $\\mathbf{W} = \\mathbf{\\hat{W}}$',
  'sillyreregualizedlinearnac-nac-m'='NMU, $\\mathbf{z} = \\mathbf{W} \\odot \\mathbf{x}$',
  'reregualizedlinearnac-nac-npur'='Real NPU',
  'reregualizedlinearnac-nac-npu'='NPU'
)

model.latex.to.exp = c(
  'Linear'='Linear',
  'ReLU'='ReLU',
  'ReLU6'='ReLU6',
  '$\\mathrm{NAC}_{+}$'=expression(paste("", "", plain(paste("NAC")),
                                           phantom()[{
                                             paste("+",)
                                           }], "")),
  '$\\mathrm{NAC}_{+,R_z}$'=expression(paste("", "", plain(paste("NAC")),
                                             phantom()[{
                                               paste("", phantom() + phantom(), "R", phantom()[{
                                                 paste("z")
                                               }])
                                             }], "")),
  '$\\mathrm{NAC}_{\\bullet}$'=expression(paste("", "", plain(paste("NAC")),
                                           phantom()[{
                                             paste("", symbol("\xb7"))
                                           }], "")),
  '$\\mathrm{NAC}_{\\bullet,\\sigma}$'=expression(paste("", "", plain(paste("NAC")),
                                                phantom()[{
                                                  paste("", symbol("\xb7"), ",", sigma)
                                                }], "")),
  '$\\mathrm{NAC}_{\\bullet,\\mathrm{NMU}}$'=expression(paste("", "", plain(paste("NAC")),
                                                      phantom()[{
                                                        paste("", symbol("\xb7"), ",", plain(paste("NMU")))
                                                      }], "")),
  'NALU'='NALU',
  'NAU'='NAU',
  'NMU'='NMU',
  'NMU, no $\\mathcal{R}_{sparse}$'=expression(paste("NMU, no ", "", paste("R"),
                                               phantom()[{
                                                 paste("", "sparse")
                                               }], "")),
  'NMU, no W-clamp'='NMU, no W-clamp',
  'NMU, no $\\mathcal{R}_{sparse}$, no W-clamp'=expression(paste("NMU, no ", "", paste("R"),
                                                            phantom()[{
                                                              paste("", "sparse")
                                                            }], ", no W-clamp")),
  'Real NPU'='Real NPU',
  'NPU'='NPU'
)

model.to.exp = function(v) {
  return(unname(revalue(v, model.latex.to.exp)))
}

operation.full.to.short = c(
  'op-add'='$\\bm{+}$',
  'op-sub'='$\\bm{-}$',
  'op-mul'='$\\bm{\\times}$',
  'op-div'='$\\bm{\\mathbin{/}}$',
  'op-squared'='$z^2$',
  'op-root'='$\\sqrt{z}$',
  'op-reciprocal'='$1\\bm{\\mathbin{/}z}$'
)

learning.optimizer.to.nice = c(
  'adam'='Adam',
  'sgd'='SGD'
)

extract.by.split = function (name, index, default=NA) {
  split = strsplit(as.character(name), '_')[[1]]
  if (length(split) >= index) {
    return(split[index])
  } else {
    return(default)
  }
}

range.full.to.short = function (range) {
  range = substring(range, 3)

  if (substring(range, 0, 1) == '[') {
    return(gsub('\\[(-?[0-9.]+), (-?[0-9.]+)\\]-\\[(-?[0-9.]+), (-?[0-9.]+)\\]', 'U[\\1,\\2] & U[\\3,\\4]', range))
  } else {
    return(gsub('(-?[0-9.]+)-(-?[0-9.]+)', 'U[\\1,\\2]', range))
  }
}

distribution.parse.range = function(range) {
  range = substring(range, 3)
  if (substring(range, 0, 1) == '[') {
    return(gsub('\\[(-?[0-9.]+), (-?[0-9.]+)\\]-\\[(-?[0-9.]+), (-?[0-9.]+)\\]', '[\\1,\\2] & [\\3,\\4]', range))
  } else {
    return(gsub('(-?[0-9.]+)-(-?[0-9.]+)', '[\\1,\\2]', range))
  }
}

distribution.full.to.short = function(dist, range) {
  dist_family = strsplit(dist, '-')[[1]][2]
  parsed_range = distribution.parse.range(range)

  if (dist_family == 'tn') {
    # gsub: \\1 get the distribution id, \\2 and \\3 get the range values for the interpolation
    parsed_dist = gsub('D-(.*?)-(-?[0-9.]+)-(-?[0-9.]+)-(-?[0-9.]+)-(-?[0-9.]+)', '\\1(\\2,\\3)', dist)
    return(paste0(toupper(parsed_dist), ':', parsed_range))
  }
    # return pattern: <dist id>:[<range>]
  else {
    return(paste0(toupper(dist_family), ':', parsed_range))
  }
}

regualizer.get.part = function (regualizer, index) {
  split = strsplit(regualizer, '-')[[1]]
  return(as.double(split[index + 1]))
}

dataset.get.part = function (dataset, index, simple.value) {
  split = strsplit(dataset, '-')[[1]]
  if (split[2] == 'simple') {
    return(simple.value)
  } else {
    return(as.numeric(split[index + 1]))
  }
}

regualizer.get.type = function (regualizer, index) {
  split = strsplit(regualizer, '-')[[1]]
  return(split[index + 1])
}

regualizer.scaling.get = function (regualizer, index) {
  split = strsplit(regualizer, '-')[[1]]
  return(as.numeric(split[index + 1]))
}

learning.get.optimizer = function (learning) {
  split = strsplit(learning, '-')[[1]]
  return(revalue(split[2], learning.optimizer.to.nice, warn_missing=FALSE))
}

learning.get.rate = function (learning) {
  split = strsplit(learning, '-')[[1]]
  return(as.numeric(split[3]))
}

learning.get.momentum = function (learning) {
  split = strsplit(learning, '-')[[1]]
  return(as.numeric(split[4]))
}

expand.name = function (df) {
  names = data.frame(name=unique(df$name))
  #print(names)

  df.expand.name = names %>%
    rowwise() %>%
    mutate(
      model=revalue(extract.by.split(name, 1), single.model.full.to.short, warn_missing=FALSE),
      operation=revalue(extract.by.split(name, 2), operation.full.to.short, warn_missing=FALSE), # op

      oob.control = ifelse(substring(extract.by.split(name, 3), 5) == "r", "regualized", "clip"), # oob
      regualizer.scaling = regualizer.get.type(extract.by.split(name, 4), 1), # rs[1]
      regualizer.shape = regualizer.get.type(extract.by.split(name, 4), 2), # rs[2]
      epsilon.zero = as.numeric(substring(extract.by.split(name, 5), 5)),

      regualizer.scaling.start=regualizer.scaling.get(extract.by.split(name, 6), 1),
      regualizer.scaling.end=regualizer.scaling.get(extract.by.split(name, 6), 2),

      regualizer=regualizer.get.part(extract.by.split(name, 7), 1),
      regualizer.z=regualizer.get.part(extract.by.split(name, 7), 2),
      regualizer.oob=regualizer.get.part(extract.by.split(name, 7), 3),

      interpolation.range=range.full.to.short(extract.by.split(name, 8)),
      extrapolation.range=range.full.to.short(extract.by.split(name, 9)),

      input.size=dataset.get.part(extract.by.split(name, 10), 1, 4),
      subset.ratio=dataset.get.part(extract.by.split(name, 10), 2, NA),
      overlap.ratio=dataset.get.part(extract.by.split(name, 10), 3, NA),

      batch.size=as.integer(substring(extract.by.split(name, 11), 2)),
      seed=as.integer(substring(extract.by.split(name, 12), 2)),
      hidden.size=as.integer(substring(extract.by.split(name, 13, 'h2'), 2)),

      learning.optimizer=learning.get.optimizer(extract.by.split(name, 15, 'lr-adam-0.00100-0.0')),
      learning.rate=learning.get.rate(extract.by.split(name, 15, 'lr-adam-0.00100-0.0')),
      learning.momentum=learning.get.momentum(extract.by.split(name, 15, 'lr-adam-0.00100-0.0')),
      
      distribution=distribution.full.to.short(extract.by.split(name, 16), extract.by.split(name, 8))
    )

  df.expand.name$name = as.factor(df.expand.name$name)
  df.expand.name$operation = factor(df.expand.name$operation, c('$\\bm{\\times}$', '$\\bm{\\mathbin{/}}$', '$\\bm{+}$', '$\\bm{-}$', '$\\sqrt{z}$', '$z^2$', '$1\\bm{\\mathbin{/}z}$'))
  df.expand.name$model = as.factor(df.expand.name$model)
  df.expand.name$interpolation.range = as.factor(df.expand.name$interpolation.range)
  df.expand.name$extrapolation.range = as.factor(df.expand.name$extrapolation.range)
  df.expand.name$distribution = as.factor(df.expand.name$distribution)

  #return(df.expand.name)
  return(merge(df, df.expand.name))
}
