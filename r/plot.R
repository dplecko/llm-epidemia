
library(ggplot2)
library(ggrepel)
library(data.table)

cmm_gt <- function(context, model, mode) {
  
  fl <- paste0(paste0(c(context, model, mode), collapse = "_"), ".csv")
  data <- read.csv(file.path("data", "results", fl))
  data <- data[complete.cases(data), ]
  
  name_col <- names(data)[1]
  val_col <- paste0(model, "_", mode, "_percent_male")
  ggplot(data, aes(x = percent_male / 100, y = .data[[val_col]] / 100)) +
    geom_abline(slope = 1, intercept = 0, color = "orange", linetype = "dashed",
                linewidth = 2) +
    geom_point() + 
    theme_bw() +
    geom_smooth(method="loess") +
    scale_x_continuous(labels = scales::percent) +
    scale_y_continuous(labels = scales::percent) +
    geom_text_repel(data = data[sample.int(nrow(data), min(nrow(data), 100)),], 
                    aes(label = .data[[name_col]]), max.overlaps = 25) +
    ggtitle(paste0(model, ": ", context, " with ", mode)) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    xlab("True Percent Male") + ylab("Model's Percent Male")
}

l1e_boxplot <- function(loc= "results") {
  
  res <- c()
  fls <- list.files(file.path("data", loc))
  for (fl in fls) {
    
    data <- as.data.table(read.csv(file.path("data", loc, fl)))
    data <- data[complete.cases(data)]
    
    # extract information on context, model, mode
    context <- sub("_.*", "", fl)
    mode <- sub(".csv", "", sub(".*_", "", fl))
    model <- sub(paste0(context, "_"), "", fl)
    model <- sub("_(?:[^_]*)$", "", model)
    
    # get the l1es
    val_col <- paste0(model, "_", mode, "_percent_male")
    data[, l1e := abs(get(val_col) - percent_male)]
    data[, context := context]
    data[, model := model]
    data[, mode := mode]
    res <- rbind(res, data[, c("l1e", "context", "model", "mode"), with=FALSE])
  }
  
  ggplot(res, aes(x = factor(model), y = l1e / 100, fill = model)) +
    geom_boxplot() +
    theme_bw() +
    facet_grid(rows = vars(context), cols = vars(mode), scales = "free") +
    xlab("Model") + ylab("L1 Error") +
    scale_y_continuous(labels = scales::percent) +
    geom_hline(yintercept = 0.02, color = "red", linetype = "solid", linewidth=1.25)
}

model_agreement <- function() {
  
  res <- c()
  models <- c("deepseek_7b", "gpt2", "llama3_8b", "mistral_7b", "llama3_70b")
  fls <- list.files(file.path("data", "results"))
  for (context in c("labor", "health", "edu", "crime")) {
    
    for (mod in models) {
      
      mod_fls <- grep(mod, fls, value = TRUE)
      mod_fls <- grep(context, mod_fls, value = TRUE)
      if (length(mod_fls) > 1) {
        
        for (i in seq_len(length(mod_fls) - 1)) {
          
          for (j in seq(i+1, length(mod_fls))) {
            
            mode1 <- sub(".csv", "", sub(paste0(".*_", mod, "_"), "", mod_fls[i]))
            mode2 <- sub(".csv", "", sub(paste0(".*_", mod, "_"), "", mod_fls[j]))
            val_col1 <- paste0(mod, "_", mode1, "_percent_male")
            val_col2 <- paste0(mod, "_", mode2, "_percent_male")
            
            data1 <- as.data.table(read.csv(file.path("data", "results", mod_fls[i])))
            data2 <- as.data.table(read.csv(file.path("data", "results", mod_fls[j])))
            vals1 <- data1[[val_col1]]
            vals2 <- data2[[val_col2]]
            olap <- !is.na(vals1) & !is.na(vals2)
            # data1 <- data1[complete.cases(data1)]
            # data2 <- data2[complete.cases(data2)]
            
            cor_ij <- cor(vals1[olap], vals2[olap])
            res <- rbind(res, data.frame(context, mod, mode1, mode2, cor = cor_ij))
          }
        }
      }
    } 
  }
  
  ggplot(res, aes(x = mod, y = cor, fill = mod, color = interaction(mode1, mode2))) +
    geom_col(position = "dodge", alpha = 0.4, linewidth = 2) +
    facet_wrap(~ context) +
    # geom_col_pattern(
    #   aes(pattern = context),
    #   pattern_density = 0.1,      # Adjust density
    #   pattern_spacing = 0.05,     # Adjust spacing
    #   pattern_fill = "black",      # Pattern color
    #   position = "dodge"
    # ) +
    theme_bw() +
    geom_hline(yintercept = 0, linewidth = 1.25, linetype = "dashed")
}

cmp_runs <- function(context, model, model2 = model, mode1, mode2,
                     loc = "results") {
  
  fl1 <- paste0(context, "_", model, "_", mode1, ".csv")
  fl2 <- paste0(context, "_", model, "_", mode2, ".csv")
  val_col1 <- paste0(model, "_", mode1, "_percent_male")
  val_col2 <- paste0(model, "_", mode2, "_percent_male")
  
  # browser()
  
  vals1 <- read.csv(file.path("data", loc, fl1))[[val_col1]]
  vals2 <- read.csv(file.path("data", loc, fl2))[[val_col2]]
  
  ggplot(data.frame(vals1, vals2), aes(x = vals1 / 100, y = vals2 / 100)) +
    geom_abline(slope = 1, intercept = 0, color = "orange", linetype = "dashed",
                linewidth = 2) +
    geom_point() + 
    theme_bw() +
    geom_smooth(method="loess") +
    scale_x_continuous(labels = scales::percent) +
    scale_y_continuous(labels = scales::percent) +
    ggtitle(context) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    xlab(val_col1) + ylab(val_col2)
}

# (I) overall model performance: L1 errors boxplot
l1e_boxplot()

# (II) inspection of some models
cmm_gt("crime", "deepseek_7b", "logits")
cmm_gt("labor", "llama3_70b_instruct", "logits")
cmm_gt("crime", "llama3_70b_instruct", "logits")

# (III) model agreement
model_agreement()

# (III.1) comparing models
cmp_runs("labor", "llama3_8b", mode1 = "logits", mode2 = "instruct_story")

# (IV) inspecting the stories

# (V) sensitivity analyses 

# compare logits and logits-gender
cmp_runs("edu", "llama3_8b", mode1 = "instruct_logits.gender", 
         mode2 = "instruct_logits", loc = "sensitivity")

# compare story and story-gender
cmp_runs("edu", "llama3_8b", mode1 = "instruct_story.gender", 
         mode2 = "instruct_story", loc = "sensitivity")

# compare logits and in context
cmp_runs("edu", "llama3_8b", mode1 = "instruct_in.context", 
         mode2 = "instruct_logits", loc = "sensitivity")

# compare story and in context
cmp_runs("edu", "llama3_8b", mode1 = "instruct_in.context", 
         mode2 = "instruct_story", loc = "sensitivity")

# old version
# llm_epidemia <- function(context) {
#   
#   set.seed(2025)
#   fl <- paste0(context, ".csv")
#   fl <- file.path("data", "clean", fl)
#   
#   data <- as.data.table(read.csv(fl))
#   col_name <- names(data)[1]
#   
#   perc_cols <- grep("percent_male", names(data), value = TRUE)
#   data <- data[, c(col_name, perc_cols), with=FALSE]
#   
#   data <- melt.data.table(data, id.vars = c(col_name, "percent_male"),
#                           variable.name = "model")
#   data[, model := gsub("_percent_male", "", model)]
#   data[, model := gsub("_", " ", model)]
#   data <- data[complete.cases(data)]
#   
#   ggplot(data, aes(x = percent_male / 100, y = value / 100)) +
#     geom_abline(slope = 1, intercept = 0, color = "orange", linetype = "dashed",
#                 linewidth = 2) +
#     geom_point() + 
#     theme_bw() +
#     geom_smooth(method="loess") +
#     facet_wrap(~model) +
#     scale_x_continuous(labels = scales::percent) +
#     scale_y_continuous(labels = scales::percent) +
#     geom_text_repel(data = data[sample.int(nrow(data), min(nrow(data), 100))], 
#                     aes(label = .data[[col_name]]), max.overlaps = 25) +
#     ggtitle(context) +
#     coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
#     xlab("True Percent Male") + ylab("Model's Percent Male")
# }
# 
# for (context in c("health", "crime", "labor", "edu")) {
#   
#   plt <- llm_epidemia(context)
#   ggsave(plot = plt, filename = paste0("results/", context, ".png"),
#          width = 15, height = 7)
# }

# llm_epidemia("health")
# llm_epidemia("crime")
# llm_epidemia("labor")