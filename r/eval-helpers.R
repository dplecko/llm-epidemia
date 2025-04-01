
#' Weighted Kolmogorov–Smirnov Statistic
#'
#' Computes the weighted KS statistic between two numeric vectors.
#'
#' @param x Numeric vector.
#' @param y Numeric vector.
#' @param w_x Optional weights for \code{x}. Defaults to uniform weights.
#' @param w_y Optional weights for \code{y}. Defaults to uniform weights.
#'
#' @return A single numeric value representing the KS statistic.
ks_w <- function(x, y, w_x = NULL, w_y = NULL) {
  if (is.null(w_x)) w_x <- rep(1, length(x))
  if (is.null(w_y)) w_y <- rep(1, length(y))
  
  dt <- rbind(
    data.table(val = x, w = w_x, grp = "x"),
    data.table(val = y, w = w_y, grp = "y")
  )
  
  setorder(dt, val)
  dt[, wx_cum := cumsum(ifelse(grp == "x", w, 0))]
  dt[, wy_cum := cumsum(ifelse(grp == "y", w, 0))]
  
  total_wx <- sum(w_x)
  total_wy <- sum(w_y)
  
  dt[, cdf_x := wx_cum / total_wx]
  dt[, cdf_y := wy_cum / total_wy]
  
  dt[, stat := abs(cdf_x - cdf_y)]
  
  return(max(dt$stat))
}

#' Weighted Tabulation
#'
#' Tabulates weighted counts for integer values in a vector.
#'
#' @param x Integer vector (positive values only).
#' @param w Optional numeric weights. Defaults to equal weights.
#' @param nbins Number of bins for tabulation. Defaults to \code{max(x)}.
#'
#' @return A numeric vector of length \code{nbins} with weighted counts.
tabulate_w <- function(x, w = NULL, nbins = max(x)) {
  if (is.null(w)) w <- rep(1, length(x))
  out <- numeric(nbins)
  for (i in seq_along(x)) {
    xi <- as.integer(x[i])
    if (xi > 0 && xi <= nbins) {
      out[xi] <- out[xi] + w[i]
    }
  }
  out
}

#' Convert Categorical Values to Weighted Distribution
#'
#' Maps a sample (which is categorical factor) to a probability mass function.
#'
#' @param x Categorical vector representing the sample.
#' @param w Numeric weights corresponding to \code{x}.
#' @param nbins Integer specifying number of bins in the output distribution.
#'
#' @return A numeric vector of length \code{nbins} summing to 1.
cat_to_distr <- function(x, w, nbins) {
  
  x <- 1 + as.numeric(x)
  distr <- tabulate_w(x, w, nbins)
  distr / sum(distr)
}