
load_difftime.nhanes_tbl <- function(x, rows, cols = colnames(x),
                                  id_hint = id_vars(x),
                                  time_vars = ricu::time_vars(x), ...) {
  
  sec_as_mins <- function(x) ricu:::min_as_mins(as.integer(x / 60))
  ricu:::warn_dots(...)
  ricu:::load_eiau(x, {{ rows }}, cols, id_hint, time_vars, sec_as_mins)
}

library(ricu)
library(fst)
library(data.table)
library(mice)