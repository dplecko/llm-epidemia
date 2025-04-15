
model_name <- function(mod) {
  
  if (length(mod) > 1) return(sapply(mod, model_name))
  
  switch (mod,
    llama3_8b_instruct = "LLama3 8B",
    llama3_70b_instruct = "LLama3 70B",
    mistral_7b_instruct = "Mistral 7B",
    deepseek_7b_chat = "DeepSeek 7B",
    phi4 = "Phi4",
    gemma3_27b_instruct = "Gemma3 27B",
    nhanes = "NHANES",
    gss = "GSS"
  )
}

model_unname <- function(mod) {
  
  if (length(mod) > 1) return(sapply(mod, model_name))
  
  switch (mod,
          `LLama3 8B` = "llama3_8b_instruct",
          `LLama3 70B` = "llama3_70b_instruct",
          `Mistral 7B` = "mistral_7b_instruct",
          `DeepSeek 7B` = "deepseek_7b_chat",
          `Phi4` = "phi4",
          `Gemma3 27B` = "gemma3_27b_instruct",
          `NHANES` = "nhanes",
          `GSS` = "gss"
  )
}

dat_name_clean <- function(x) {
  
  sub("\\.csv$|\\.parquet$", "", tail(strsplit(x, split = "/")[[1]], n = 1))
}

sync_bench <- function() {
  
  system("rsync -avz --update --progress -e ssh eb0:~/llm-epidemia/data/benchmark/ ~/trust/llm-epidemia/data/benchmark/")
}
