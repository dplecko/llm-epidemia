fl <- "data/benchmark/nhanes_sample_census_age.json"
res <- jsonlite::read_json(fl)[[1]]

df <- data.frame(vals = as.numeric(res$true_vals), 
                 wgh = as.numeric(res$weights), dataset = "census")
df <- df[df$vals > 18, ]

df <- rbind(
  df,
  data.frame(
    vals = as.numeric(res$model_vals), wgh = 1, dataset = "nhanes"
  )
)

ggplot(df, aes(x = vals, weight = wgh, fill = dataset)) +
  geom_density(alpha = 0.5) + theme_bw() +
  xlab("Age") + ylab("Density") + 
  ggtitle("Age distribution in NHANES and Census") +
  theme(legend.position = "bottom")

eval_task("nhanes", task_specs[[1]])
