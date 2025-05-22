
source("r/eval.R")
source("r/eval-helpers.R")
source("r/helpers.R")
source("r/zzz-deps.R")
source_python("workspace/task_spec.py")

task_to_filename <- function(model_name, task_spec) {
  dataset_name <- basename(tools::file_path_sans_ext(task_spec$dataset))
  
  if ("v_cond" %in% names(task_spec)) {
    cond_vars_str <- paste(task_spec$v_cond, collapse = "_")
    file_name <- sprintf("%s_%s_%s_%s.parquet", model_name, dataset_name, task_spec$v_out, cond_vars_str)
  } else {
    file_name <- sprintf("%s_%s_%s", model_name, dataset_name, task_spec$variables[[1]])
    
    if (length(task_spec$variables) > 1) {
      file_name <- sprintf("%s_%s", file_name, task_spec$variables[[2]])
    }
    
    file_name <- paste0(file_name, ".json")
  }
  
  return(file_name)
}

plt_bin <- function(model_name, task) {
  
  fl <- task_to_filename(model_name, task)
  res <- jsonlite::read_json(file.path("data", "benchmark", fl))
  
  df <- NULL
  for (i in seq_along(res)) {
    
    if (length(res[[i]]$model_weights) > 2) {
      cat("Not Binary\n")
      return(NULL)
    }
    # browser()
    tr <- try(res[[i]]$true_weights[[2]])
    if (class(tr) == "try-error") browser()
    p_true <- res[[i]]$true_weights[[2]] / sum(unlist(res[[i]]$true_weights))
    p_mod <- res[[i]]$model_weights[[2]]
    df <- rbind(df, data.frame(p_true, p_mod, cond = res[[i]]$condition,
                               model = model))
  }
  
  df
}

models <- c("llama3_8b_instruct", "llama3_70b_instruct",
            "phi4", "gemma3_27b_instruct", "mistral_7b_instruct")

df_plt <- c()
for (model in models) {
  df_plt <- rbind(df_plt, df <- plt_bin(model, task_specs[[13]]))
}


ggplot(df_plt, aes(x = p_true, y = p_mod)) +
  geom_point() + geom_smooth() + theme_bw() +
  geom_abline(intercept = 0, slope = 1, color = "orange", linewidth=1.25) +
  facet_wrap(~model) +
  coord_cartesian(xlim=c(0, 1), ylim=c(0,1))
