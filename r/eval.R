
#' Evaluate a Task and Dispatch to Type-Specific Evaluation
#'
#' Determines whether the task is binary, categorical, or continuous, 
#' reads results from disk, and dispatches to the appropriate evaluation function.
#'
#' @param model_name Name of the model used.
#' @param task A list describing the task, including dataset, variables, levels, and mode.
#'
#' @return A data.frame containing the benchmarking results with score for each
#' level of the conditioning variable.
eval_task <- function(model_name, task) {
  
  mode <- task$mode
  dataset <- dat_name_clean(task$dataset)
  v1 <- task$variables[1]
  v2 <- task$variables[2]
  levels <- task$levels
  
  fl <- paste(c(model_name, mode, dataset, v1, v2), collapse = "_")
  fl <- paste0(fl, ".json")
  res <- jsonlite::read_json(file.path("data", "results", "benchmark", fl))
  
  if (length(task$levels) == 2) {
    
    return(eval_bin(res, model_name, mode, dataset, v1, v2))
  } else if (!is.null(task$levels)) {
    
    return(eval_cat(res, model_name, mode, dataset, v1, v2, levels))
  } else {
    
    return(eval_cts(res, model_name, mode, dataset, v1, v2))
  }
}

#' Evaluate Continuous Predictions
#'
#' Computes KS-based benchmarking score for continuous outcome predictions.
#'
#' @param res List of model output and true values for levels of conditioning variable.
#' @param model_name Name of the model.
#' @param mode Evaluation mode ("logits", "sample", etc.).
#' @param dataset Name of the dataset.
#' @param v1 Name of the outcome variable.
#' @param v2 Name of the conditioning variable.
#'
#' @return A data.frame with mean true and model values, and benchmarking score 
#' for each level of the conditioning variable.
eval_cts <- function(res, model_name, mode, dataset, v1, v2) {

  df <- c()
  for (i in seq_along(res)) {
    
    n_mc <- length(res[[i]]$model_vals)
    val_true <- as.numeric(res[[i]]$true_vals)
    wgh_true <- as.numeric(res[[i]]$weights)
    if (length(wgh_true) == 0) wgh_true <- rep(1, length(val_true))
    val_mod <- as.numeric(unlist(res[[i]]$model_vals))
    
    best_err <- c()
    for (boot in seq_len(100)) {
      
      b_idx <- sample(1:length(val_true), size = n_mc, 
                      prob = wgh_true, replace = TRUE)
      val_true_bt <- val_true[b_idx]
      test_bt <- ks_w(x = val_true_bt, w_x = NULL, y = val_true, w_y = wgh_true)
      
      best_err <- c(best_err, test_bt)
    }
    
    best_err <- quantile(best_err, probs = 0.975)
    worst_bt <- runif(1000, min = min(val_true), max = max(val_true))
    worst_err <- ks_w(x = val_true, w_x = wgh_true, y = worst_bt, w_y = NULL)
    
    score <- ks_w(x = val_true, w_x = wgh_true, y = val_mod, w_y = NULL)
    
    bench <- (score - worst_err) / (best_err - worst_err)
    bench <- 100 * max(min(bench, 1), 0)
    df <- rbind(df, data.frame(p_true = sum(val_true * wgh_true) / sum(wgh_true), 
                               p_mod = mean(val_mod), bench, cond = res[[i]]$condition))
  }
  
  df
}

#' Evaluate Categorical Predictions
#'
#' Computes a benchmark score by comparing weighted distributions over levels.
#'
#' @param res List of model output and true values per condition.
#' @param model_name Name of the model.
#' @param mode Evaluation mode.
#' @param dataset Dataset name.
#' @param v1 Outcome variable name.
#' @param v2 Conditioning variable name.
#' @param levels List of grouped category levels.
#'
#' @return A data.frame of benchmarking scores, with a "distr" attribute 
#' containing proportions per categorical levels and value of conditioning variable.
eval_cat <- function(res, model_name, mode, dataset, v1, v2, levels) {
  
  if (v1 == "party") browser()
  
  nbins <- length(levels)
  df <- distr_df <- c()
  lvl_names <- sapply(levels, `[[`, 1L)
  for (i in seq_along(res)) {
    
    n_mc <- length(res[[i]]$model_vals)
    val_true <- as.numeric(res[[i]]$true_vals)
    wgh_true <- as.numeric(res[[i]]$weights)
    if (length(wgh_true) == 0) wgh_true <- rep(1, length(val_true))
    
    distr_true <- cat_to_distr(x = val_true, w = wgh_true, nbins = nbins)
    distr_mod <- cat_to_distr(x = unlist(res[[i]]$model_vals), w = NULL, nbins = nbins)
    
    distr_df <- rbind(distr_df,
      data.frame(lvl = 1:nbins, lvl_names, prop = distr_true, 
                 type = "Reality", cond = res[[i]]$condition)
    )
    
    distr_df <- rbind(distr_df,
      data.frame(lvl = 1:nbins, lvl_names, prop = distr_mod, 
                 type = "Model", cond = res[[i]]$condition)
    )

    best_err <- c()
    for (boot in seq_len(100)) {
      
      b_idx <- sample(1:length(val_true), size = n_mc, 
                      prob = wgh_true, replace = TRUE)
      val_true_bt <- val_true[b_idx]
      distr_true_bt <- cat_to_distr(x = val_true_bt, w = NULL, nbins = nbins)
      best_err <- c(best_err, abs(sum(distr_true_bt - distr_true)))
    }
    
    best_err <- quantile(best_err, probs = 0.975)
    worst_bt <- runif(1000, min = min(val_true), max = max(val_true))
    worst_err <- sum(abs(rep(1/nbins, nbins) - distr_true))
    score <- sum(abs(distr_true - distr_mod))
    
    bench <- (score - worst_err) / (best_err - worst_err)
    bench <- 100 * max(min(bench, 1), 0)
    df <- rbind(df, data.frame(bench, cond = res[[i]]$condition))
  }
  
  attr(df, "distr") <- distr_df
  df
}

#' Evaluate Binary Predictions
#'
#' Computes benchmark score for binary classification, based on absolute error.
#'
#' @param res List of model output and true values per condition.
#' @param model_name Name of the model.
#' @param mode Evaluation mode.
#' @param dataset Dataset name.
#' @param v1 Outcome variable name.
#' @param v2 Conditioning variable name.
#'
#' @return A data.frame with true and model means, benchmarking score, 
#' and value of the conditioning variable.
eval_bin <- function(res, model_name, mode, dataset, v1, v2) {
  
  df <- c()
  for (i in seq_along(res)) {
    
    n_mc <- length(res[[i]]$model_vals)
    val_true <- as.numeric(res[[i]]$true_vals)
    wgh_true <- as.numeric(res[[i]]$weights)
    if (length(wgh_true) == 0) wgh_true <- rep(1, length(val_true))
    p_true <- sum(val_true * wgh_true) / sum(wgh_true)
    p_mod <- if (n_mc == 2) res[[i]]$model_vals[[2]] else mean(do.call(c, res[[i]]$model_vals))
    
    if (n_mc == 2) n_mc <- 10^6
    best_err <- 2 * sqrt(p_true * (1-p_true) / n_mc)
    worst_err <- 1 / 2 * (p_true^2 + (1-p_true)^2)
    
    score <- abs(p_true - p_mod)
    
    bench <- (score - worst_err) / (best_err - worst_err)
    bench <- 100 * max(min(bench, 1), 0)
    df <- rbind(df, data.frame(p_true, p_mod, bench, cond = res[[i]]$condition))
  }
  
  df
}

#' Extract Overall Score from Evaluation Results
#'
#' Computes the mean benchmark score from an evaluation result.
#'
#' @param df Data.frame returned by an evaluation function.
#'
#' @return Numeric score (mean of \code{bench} column).
eval_to_score <- function(df) {
  
  mean(df$bench)
}

#' Plot Evaluation Results
#'
#' Visualizes model vs. true distributions or means depending on task type.
#'
#' @param df Evaluation data.frame returned by \code{eval_*()}.
#' @param model_name Name of the evaluated model.
#' @param task Task specification list including levels and metadata.
#'
#' @return A \code{ggplot} object for visualization.
plt_eval <- function(df, model_name, task) {
  
  mode <- task$mode
  dataset <- dat_name_clean(task$dataset)
  v1 <- task$variables[1]
  v2 <- task$variables[2]
  
  if (length(task$levels) == 2 || is.null(task$levels)) {
    
    p <- ggplot(df, aes(x = p_true, y = p_mod)) +
      geom_abline(slope = 1, intercept = 0, color = "orange", linetype = "dashed",
                  linewidth = 2) +
      geom_point() + 
      theme_bw() +
      geom_smooth(method="loess") +
      geom_text_repel(data = df[sample.int(nrow(df), min(nrow(df), 100)),], 
                      aes(label = cond), max.overlaps = 25) +
      ggtitle(task[["name"]]) +
      xlab("True Mean") + ylab("Model's Mean")
    
    if (length(task$levels) == 2) 
      p <- p + scale_x_continuous(labels = scales::percent) +
        scale_y_continuous(labels = scales::percent) +
        coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
  } else {
    
    lvls <- sapply(task$levels, `[[`, 1L)
    distr <- attr(df, "distr")
    p <- ggplot(distr, aes(x = lvl, y = prop, fill = type)) +
      geom_col(position = "dodge", color = "black") +
      facet_wrap(~ cond) +
      scale_x_continuous(breaks = 1:length(lvls), labels = lvls) +
      theme_bw() +
      scale_y_continuous(labels = scales::percent) +
      ylab("Proportion") + xlab("Level") +
      scale_fill_discrete(name = "Distribution") +
      theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  }
  
  p
}
