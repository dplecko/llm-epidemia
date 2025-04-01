
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
df <- as.data.frame(dat)
names(df) <- unlist(df[1, ])
df <- df[-1, ]

cen <- data.frame(weight=as.numeric(as.character(df$PWGTP)))
cen$state <- factor(df$state)

cen$sex <- c('male', 'female')[1+(df$SEX=='2')]
cen$sex <- factor(cen$sex)
cen$age <- as.numeric(as.character(df$AGEP))

cen$race <- c('white',
              'black',
              'AIAN',
              'AIAN',
              'AIAN',
              'asian',
              'NHOPI',
              'other',
              'mix')[as.numeric(as.character(df$RAC1P))]
cen$race <- factor(cen$race)
cen$hispanic_origin <- factor(c('no', 'yes')[1+(df$HISP!='01')])

cen$nativity <- c('native', 'foreign-born')[as.numeric(as.character(df$NATIVITY))]
cen$nativity <- as.factor(cen$nativity)

cen$citizenship <- as.numeric(as.character(df$CIT))
cen$citizenship <- factor(c('born in the US',
                            'born in Puerto Rico',
                            'born abroad of American parents',
                            'naturalized citizen',
                            'not a citizen')[cen$citizenship])
cen$marital <- c('married',
                 'widowed',
                 'divorced',
                 'separated',
                 'never married')[as.numeric(as.character(df$MAR))]
cen$marital <- factor(cen$marital)

cen$family_size <- as.numeric(as.character(df$NPF))

cen$children <- as.numeric(as.character(df$NOC))

cen$english_level <- as.numeric(as.character(df$ENG)) #4 not at all, 0 native

cen$education_level <- as.numeric(as.character(df$SCHL))
cen$education_level[cen$education_level == 0] <- NA
cen$education <- c('No schooling completed',
                   'Nursery school, preschool',
                   'Kindergarten',
                   'Grade 1',
                   'Grade 2',
                   'Grade 3',
                   'Grade 4',
                   'Grade 5',
                   'Grade 6',
                   'Grade 7',
                   'Grade 8',
                   'Grade 9',
                   'Grade 10',
                   'Grade 11',
                   '12th grade - no diploma',
                   'Regular high school diploma',
                   'GED or alternative credential',
                   'Some college, but less than 1 year',
                   '1 or more years of college credit, no degree',
                   "Associate's degree",
                   "Bachelor's degree",
                   "Master's degree",
                   "Professional degree beyond a bachelor's degree",
                   'Doctorate degree')[cen$education_level]

##########################
cen$earnings <- as.numeric(as.character(df$PERNP))
cen$salary <- as.numeric(as.character(df$WAGP))
cen$salary[cen$salary == -1] <- NA

cen$hours_worked <- as.numeric(as.character(df$WKHP))

cen$employment_status <- c(NA,
                           'employed',
                           'not at work',
                           'unemployed',
                           'employed',
                           'not at work',
                           'not in labor force')[1+as.numeric(as.character(df$ESR))]
cen$employment_status <- factor(cen$employment_status)

cen$employer <- c(NA,
                  'for-profit company',
                  'non-profit company',
                  'government',
                  'government',
                  'government',
                  'self-employed',
                  'self-employed',
                  'self-employed',
                  NA)[1+as.numeric(as.character(df$COW))]
cen$employer <- factor(cen$employer)

cen$occupation <- as.character(df$OCCP)
cen$occupation[cen$occupation=='0009'] <- NA

#convert to SOC code, the one provided in SOCP does not work but gives '*'
#whenever there is many candidates for one OCCP category
url <- 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/code_lists/ACSPUMS2023CodeLists.xls'
sheet_name <- 'Occupation'
GET(url, write_disk(tf <- tempfile(fileext = ".xls")))
occ_tab <- readxl::read_excel(tf, sheet=sheet_name, col_names=FALSE, skip=10)
occ_tab <- as.data.frame(occ_tab)

cen$occp_name <- cen$occp_code <- ''
for(i in 1:nrow(occ_tab)){
  print(i)
  key <- as.character(occ_tab[i, 1])
  if(is.na(key) || length(key)==0){
    next
  }
  cen$occp_name[cen$occupation == key] <- occ_tab[i,3]
  cen$occp_code[cen$occupation == key] <- as.character(occ_tab[i,2])
}
cen$occp_code <- factor(cen$occp_code)
cen$occupation <- NULL

cen$industry <- as.character(df$INDP)
cen$industry[cen$industry=='0169'] <- NA

url <- 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/code_lists/ACSPUMS2023CodeLists.xls'
sheet_name <- 'Industry'
GET(url, write_disk(tf <- tempfile(fileext = ".xls")))
ind_tab <- readxl::read_excel(tf, sheet=sheet_name, col_names=FALSE, skip = 20)
ind_tab <- as.data.frame(ind_tab)

cen$ind_name <- ''
cen$ind_code <- ''
for(i in 1:nrow(ind_tab)) {
  print(i)
  key <- as.character(ind_tab[i, 3])
  if(is.na(key) || length(key)==0){
    next
  }
  cen$ind_name[cen$industry == key] <- ind_tab[i,2]
  cen$ind_code[cen$industry == key] <- as.character(ind_tab[i,5])
}
cen$ind_code <- factor(cen$ind_code)
cen$industry <- NULL

cen$place_of_work <- as.character(df$POWSP)
cen$place_of_work[cen$place_of_work=='N'] <- NA
cen$place_of_work = factor(cen$place_of_work)

cen$economic_region <- ''
cen$economic_region[cen$place_of_work %in%
                       c('009', '023', '025', '033', '044', '050')] <- 'New England'
cen$economic_region[cen$place_of_work %in%
                       c('010', '011', '024', '034', '036', '042')] <- 'Mideast'
cen$economic_region[cen$place_of_work %in%
                       c('017', '018', '026', '039', '055')] <- 'Great Lakes'
cen$economic_region[cen$place_of_work %in%
                       c('019', '020', '027', '029', '031', '038', '046')] <- 'Plains'
cen$economic_region[cen$place_of_work %in%
                       c('001', '005', '012', '013', '021', '022', '028',
                         '037', '045', '047', '051', '054')] <- 'Southeast'
cen$economic_region[cen$place_of_work %in%
                       c('004', '035', '040', '048')] <- 'Southwest'
cen$economic_region[cen$place_of_work %in%
                       c('008', '016', '040', '048',
                         '030', '049', '056')] <- 'Rocky Mountain'
cen$economic_region[cen$place_of_work %in%
                       c('002', '006', '015', '032', '041', '053')] <- 'Far West'
cen$economic_region[as.numeric(as.character(cen$place_of_work)) > 56] <- 'Abroad'
cen$economic_region[is.na(cen$place_of_work)] <- NA
cen$economic_region <- factor(cen$economic_region)

ordering = c(
  'sex',
  'age',
  
  'race',
  'hispanic_origin',
  'citizenship',
  'nativity',
  
  'marital',
  'family_size',
  'children',
  'state',
  
  'education',
  'education_level',
  'english_level',
  
  'salary',
  'hours_worked',
  'employment_status',
  
  'occp_name',
  'occp_code',
  'ind_name',
  'ind_code',
  'employer',
  'place_of_work',
  'economic_region',
  
  'weight'
)

cen <- cen[, ordering]
arrow::write_parquet(cen, sink = "data/clean/census.parquet")
