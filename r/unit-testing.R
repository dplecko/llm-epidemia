
# source R scripts and Python list of tasks
library(kableExtra)
source("r/eval.R")
source("r/eval-helpers.R")
source("r/helpers.R")
source("r/zzz-deps.R")
source_python("py/task_spec.py")
tasks <- py$task_specs

scoring_unit_test <- function(model_name, task, upper = TRUE) {
  
  mode <- task$mode
  dataset <- dat_name_clean(task$dataset)
  v1 <- task$variables[1]
  v2 <- task$variables[2]
  levels <- task$levels
  
  fl <- paste(c(model_name, mode, dataset, v1, v2), collapse = "_")
  fl <- paste0(fl, ".json")
  res <- jsonlite::read_json(file.path("data", "benchmark", fl))
  
  
  # intervene to make model values a sample from the ground truth
  n_mc <- 128
  for (i in seq_along(res)) {
    
    val_true <- as.numeric(res[[i]]$true_vals)
    wgh_true <- as.numeric(res[[i]]$weights)
    
    if (upper) {
      
      mc_sample <- sample(val_true, size = n_mc, replace = TRUE, probs = wgh_true)
    } else {
      
      # random sampling based on true responses
      if (!is.null(task$levels)) {
        
        uvals <- unique(val_true)
        distr <- runif(length(task$levels))
        distr <- cumsum(distr) / sum(distr)
        mc_sample <- uvals[sample(seq_along(uvals), size = n_mc, probs = distr)]
      } else {
        
        mc_sample <- runif(n_mc, min = min(val_true), max = max(val_true))
      }
    }
    
    mc_sample <- lapply(mc_sample, function(x) x)
    res[[i]]$model_vals <- mc_sample
  }
  
  if (length(task$levels) == 2) {
    
    return(eval_bin(res, model_name, mode, dataset, v1, v2))
  } else if (!is.null(task$levels)) {
    
    return(eval_cat(res, model_name, mode, dataset, v1, v2, levels))
  } else {
    
    return(eval_cts(res, model_name, mode, dataset, v1, v2))
  }
}

model_name <- "llama3_8b_instruct"
for (upper in c(TRUE, FALSE)) {
  
  for (task_idx in c(1, 2, 6)) {
    
    scr <- eval_to_score(scoring_unit_test(model_name, tasks[[task_idx]], upper))
    cat("Upper =", upper, "; score = ", round(scr), "\n")
  }
}

# inspecting answer decoding
answer_decoding <- function(model_name, task) {
  
  mode <- task$mode
  dataset <- dat_name_clean(task$dataset)
  v1 <- task$variables[1]
  v2 <- task$variables[2]
  levels <- task$levels
  fl <- paste(c(model_name, mode, dataset, v1, v2), collapse = "_")
  fl <- paste0(fl, ".json")
  res <- jsonlite::read_json(file.path("data", "benchmark", fl))
  
  # pick a random index
  idx <- sample(seq_along(res), size = 1)
  
  # extract the values
  model_texts <- unlist(res[[idx]]$model_texts)
  
  model_vals <- lapply(res[[idx]]$model_vals, function(x) if (is.null(x)) NA else x)
  model_vals <- tail(unlist(model_vals), n = length(model_texts))
  
  tibble::tibble(
    Text = model_texts,
    Decoded = model_vals
  ) %>%
    kable(format = "html", escape = TRUE) %>%
    kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover", "condensed"))
}

answer_decoding("llama3_8b_instruct", tasks[[8]])
