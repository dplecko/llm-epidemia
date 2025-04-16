
Sys.setenv(RICU_CONFIG_PATH = file.path("config"))
Sys.setenv(RICU_SRC_LOAD = "nhanes")

files <- c(
  "DEMO_L.xpt", # demographics
  "BMX_L.xpt", # body measures
  "BPXO_L.xpt", # blood pressure
  "DR1IFF_L.xpt", # food intake
  "TCHOL_L.xpt", # cholesterol
  "HDL_L.xpt",
  "GHB_L.xpt",
  "INS_L.xpt",
  "ALQ_L.xpt",
  "KIQ_U_L.xpt",
  "SMQ_L.xpt",
  "SLQ_L.xpt",
  "DIQ_L.xpt" # diabetes
)

url_loc <- "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"

if (!dir.exists(file.path(ricu::data_dir(), "nhanes"))) {
  
  dir.create(file.path(ricu::data_dir(), "nhanes"))
}


cat("NHANES data folder initiated\n")
for (fl in files) {
  
  fl_targ <- file.path(ricu::data_dir(), "nhanes", gsub("xpt", "fst", fl))
  if (!file.exists(fl_targ)) {
    
    dt <- haven::read_xpt(paste0(url_loc, fl))
    fst::write.fst(dt, fl_targ) 
  }
}

cat("NHANES data files downloaded\n")

source("zzz-deps.R")
source("callbacks.R")

dem <- c("mec_wgh", "diet_wgh", "age", "sex", "race")
body <- c("height", "weight", "waist", "bmi")
vitals <- c("diastolic_bp", "systolic_bp", "pulse")
blood <- c("cholesterol", "hdl", "hb_a1c", "insulin")
chronic <- c("diabetes", "alcohol_weekly", "smoking", "sleep_hours",
             "kidney_failure")
nutr <- c("kcal", "protein", "carbs", "sugar", "fat")
vars <- c(
  dem, body, vitals, blood, chronic, nutr
)

dat <- load_concepts(vars, "nhanes")
dat <- dat[mec_wgh > 0 & age >= 18]
dat$SEQN <- NULL

# 0 weights for NAs in diet_wgh
dat[is.na(diet_wgh), diet_wgh := 0]

# what is the NA pattern?
colMeans(is.na(dat))

set.seed(2025)
dat <- as.data.table(complete(mice(dat, m = 1)))

# BMI bin
dat[, bmi_bin := factor(
  .bincode(bmi, breaks = c(-Inf, 18.5, 25, 30, 35, 40, Inf)),
  labels = c("underweight", "normal", "overweight",
             "class I obese", "class II obese", "class III obese")
)]

# age group
dat[, age_group := factor(
  .bincode(age, breaks = c(-Inf, 30, 40, 50, 60, 70, Inf)),
  labels = c("18-30 years", "30-40 years", "40-50 years",
             "50-60 years", "60-70 years", "70+ years")
)]

arrow::write_parquet(as.data.frame(dat), sink = "nhanes.parquet")
cat("NHANES dataset created in parquet\n")
