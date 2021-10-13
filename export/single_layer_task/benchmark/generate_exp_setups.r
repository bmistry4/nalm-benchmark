rm(list = ls())
# setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)

# ( '[-40,-20]' '[-6,-2]' '[-6.1,-1.2]' '[-2,-0.2]'  '[[-6,-2],[2,6]]' '[0.2,2]'   '[2,6]' '[1.2,6]'  '[20,40]' )
cases = rbind()
for (operation in c('add', 'sub', 'mul', 'div')){
  cases = rbind(cases,
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=-40, range.b=-20, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=-6, range.b=-2, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=-6.1, range.b=-1.2, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=-2, range.b=-0.2, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=2, range.b=6, range.mirror=T),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=0.2, range.b=2, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=2, range.b=6, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=1.2, range.b=6, range.mirror=F),
    c(parameter='extrapolation.range', operation=operation, simple=F, input.size=2, subset.ratio=0.5, overlap.ratio=0, range.a=20, range.b=40, range.mirror=F)
  )
}

eps = data.frame(rbind(
  c(operation='mul', epsilon=0.00001),
  c(operation='add', epsilon=0.00001),
  c(operation='sub', epsilon=0.00001),
  c(operation='div', epsilon=0.00001),
  c(operation='squared', epsilon=0.00001),
  c(operation='root', epsilon=0.00001),
  c(operation='reciprocal', epsilon=0.00001)
))

dat = data.frame(cases) %>%
  merge(eps) %>%
  mutate(
    simple=as.logical(as.character(simple)),
    input.size=as.integer(as.character(input.size)),
    subset.ratio=as.numeric(as.character(subset.ratio)),
    overlap.ratio=as.numeric(as.character(overlap.ratio)),
    range.a=as.numeric(as.character(range.a)),
    range.b=as.numeric(as.character(range.b)),
    range.mirror=as.logical(as.character(range.mirror)),
    epsilon=as.numeric(as.character(epsilon))
  ) %>%
  rowwise() %>%
  mutate(
    extrapolation.range=ifelse(range.mirror, ifelse(range.b<=0, paste0('U[',range.b*-1,',',range.a*-1,'] & U[',range.a,',',range.b,']') , paste0('U[-',range.b,',-',range.a,'] & U[',range.a,',',range.b,']')), paste0('U[',range.a,',',range.b,']')), # if a and b are -ve then want: [[+,+],[-,-]]
    operation=paste0('op-', operation)
  )
  
write.csv(dat, file="exp_setups.csv", row.names=F)

