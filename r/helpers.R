
model_name <- function(mod) {
  
  if (length(mod) > 1) return(sapply(mod, model_name))
  
  switch (mod,
    llama3_8b_instruct = "LLama3 8B",
    llama3_70b_instruct = "LLama3 70B"
  )
}

model_unname <- function(mod) {
  
  if (length(mod) > 1) return(sapply(mod, model_name))
  
  switch (mod,
          `LLama3 8B` = "llama3_8b_instruct",
          `LLama3 70B` = "llama3_70b_instruct"
  )
}

dat_name_clean <- function(x) {
  
  gsub(".csv", "", tail(strsplit(x, split = "/")[[1]], n = 1))
}
