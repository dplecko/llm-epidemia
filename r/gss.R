
library(tidyverse)
library(haven)
library(survey)

# Read GSS 2022 dataset
dt <- read_dta("data/raw/gss/GSS2022.dta")

# Select relevant variables
gss_sel <- dt %>%
  select(
    # Demographics
    sex, age, race, educ, degree, income, 
    
    # Social Trust
    # trust, fair, helpful, confed, confinan, conpress,
    
    # Political & Economic Attitudes
    polviews, partyid,
    
    # Survey weight
    wtssps, wtssnrps
  )


gss_sel <- gss_sel[!is.na(gss_sel$wtssnrps), ]
sort(colMeans(is.na(gss_sel))) * 100

# 
gss <- data.frame(wgh = as.numeric(gss_sel$wtssnrps))

# clean the sex variable
gss$sex <- c("male", "female")[gss_sel$sex]

# clean age variable
gss$age <- as.numeric(gss_sel$age)

# race
gss$race <- c("white", "black", "other")[gss_sel$race]

# degree
gss$degree <- c("no high school", "high school", "junior college",
                "bachelor", "graduate")[gss_sel$degree+1]

# education
gss$edu <- as.numeric(gss_sel$educ)

# income

# political views
gss$view <- c("liberal", "liberal", "liberal", "moderate",
               "conservative", "conservative", "conservative")[gss_sel$polviews]

# party affiliation
gss$party <- c("democrat", "democrat", "democrat", "independent",
               "republican", "republican", "republican", NA)[gss_sel$partyid+1]

for (var in names(gss)) {
  
  if (is.character(gss[[var]]))
    gss[[var]] <- as.factor(gss[[var]])
}

gss_impute <- complete(mice(gss, m = 1))

write.csv(gss_impute, file = "data/clean/gss.csv")


