
library(tidycensus)

# Set your Census API key
census_api_key("1ddb0b9f7331dd07bcf3fc8b492d66c69d751cc4", overwrite = TRUE)

# Define variables
vars <- c("PWGTP", "ST", "SEX", "AGEP", "RAC1P", "HISP", "NATIVITY", "CIT", "MAR", "NPF", "NOC", "ENG", "SCHL", "PERNP", "WAGP", "WKHP", "WKW", "ESR", "COW", "OCCP", "INDP", "POWSP")
httr::GET("https://api.census.gov/data/2023/acs/acs1?get=PWGTP,ST,SEX,AGEP,RAC1P,HISP,NATIVITY,CIT,MAR,NPF,NOC,ENG,SCHL,PERNP,WAGP,WKHP,WKW,ESR,COW,OCCP,INDP,POWSP&for=state:*&key=1ddb0b9f7331dd07bcf3fc8b492d66c69d751cc4")

vars <- c("PWGTP", "ST", "SEX", "AGEP", "RAC1P", "HISP", "NATIVITY", "CIT", 
          "MAR", "NPF", "NOC", "ENG", "SCHL", "PERNP", "WAGP", "WKHP", "WKW", 
          "ESR", "COW", "OCCP", "INDP", "POWSP")

acs_data <- get_acs(geography = "state", variables = vars, year = 2023, 
                    survey = "acs1", show_call = TRUE)

# https://api.census.gov/data/2020/acs/acs5?get=NAME&for=state:36&key=911133d4eda646e6156268ac41e32d5baea2a977

write.csv(faircause::gov_census, file = "data/clean/census.csv")

library(httr)
library(jsonlite)

url <- "https://api.census.gov/data/2023/acs/acs1/pums"
vars <- c("PWGTP", "SEX", "AGEP", "RAC1P", "HISP", "NATIVITY", "CIT", "MAR", 
          "NPF", "NOC", "ENG", "SCHL", "PERNP", "WAGP", "WKHP", 
          "ESR", "COW", "OCCP", "INDP", "POWSP")
key <- "1ddb0b9f7331dd07bcf3fc8b492d66c69d751cc4"

req <- GET(url, query = list(
  get = paste(vars, collapse = ","),
  `for` = "state:*",
  key = key
))

dat <- fromJSON(content(req, as = "text"))
df <- as.data.frame(dat[-1, ])
names(df) <- dat[1, ]
