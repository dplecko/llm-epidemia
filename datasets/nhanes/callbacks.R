
nhanes_sex <- function(x, val_var, ...) {
  
  x[, c(val_var) := ifelse(get(val_var) == 1, "Male", "Female")]
}

nhanes_smoke <- function(x, val_var, ...) {
  
  x[!(get(val_var) %in% c(0, 1)), c(val_var) := 0]
  x[, c(val_var) := ifelse(get(val_var) %in% c(1, 2), 1, 0)]
}

nhanes_alcohol <- function(x, val_var, ...) {
  
  x[is.na(get(val_var)), c(val_var) := 0]
  x[, c(val_var) := ifelse(get(val_var) %in% c(1, 2, 3, 4, 5), 1, 0)]
}

nhanes_kidney <- function(x, val_var, ...) {
  
  x[, c(val_var) := ifelse(get(val_var) == 1, 1, 0)]
}

nhanes_race <- function(x, val_var, ...) {

  x[, c(val_var) := fcase(
    get(val_var) %in% c(1, 2), "Hispanic",
    get(val_var) == 3, "White",
    get(val_var) == 4, "Black",
    get(val_var) == 6, "Asian",
    get(val_var) == 7, "Other",
    default = NA_character_
  )]
}

nhanes_diab <- function(x, val_var, ...) {
  
  x[, c(val_var) := fcase(
    get(val_var) %in% c(1, 3), 1,
    get(val_var) == 2, 0,
    default = NA_real_
  )]
}

nhanes_kcal <- function(x, val_var, ...) {
  
  x[, c(val_var) := sum(get(val_var)), by = c(id_vars(x))]
}
