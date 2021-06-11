#loading libraries
library(data.table)
library(corrr)
library(caret)
library(mltools)
library(onehot)
library(mlbench) #for the data
library(doParallel)
library(FSelector)
library(tidyverse)
#loading the data
data_home <-
  fread("~/application_train.csv")


#data prep

nacols <- function(df) {
  colnames(df)[unlist(lapply(df, function(x)
    anyNA(x)))]
}

#number of missing values per variable
missing_value <-
  as.data.frame(sort(sapply(data_home, function(x)
    sum(is.na(
      x
    ))), decreasing = T))

#missing values percentage
colnames(missing_value)[1] <- "Missing_values_SUM"
missing_value$Percentage_missing <-
  (missing_value$Missing_values / nrow(data_home)) * 100
missing_value$Variables <- rownames(missing_value)

#Variables containing NAs less than 60% - 60% compromise between losing information and biased dataset
missing_value <-
  subset(missing_value, missing_value$Percentage_missing <= 60)
less_na_columns_train <-
  missing_value$Variables
less_na_columns_train

#keeping those variables
na_omitted_data <- data_home[, mget(less_na_columns_train)]

#general feature engineering to combine features in order to gain more information
#calculating percentage missing per row
feature_engineering <- na_omitted_data
feature_engineering$DAYS_EMPLOYED = ifelse(feature_engineering$DAYS_EMPLOYED == 365243,
                                           NA,
                                           feature_engineering$DAYS_EMPLOYED)
feature_engineering$INCOME_PER_PERSON = log(feature_engineering$AMT_INCOME_TOTAL / feature_engineering$CNT_FAM_MEMBERS)
feature_engineering$LOAN_INCOME_RATIO = feature_engineering$AMT_CREDIT / feature_engineering$AMT_INCOME_TOTAL
feature_engineering$ANNUITY_LENGTH = feature_engineering$AMT_CREDIT / feature_engineering$AMT_ANNUITY
feature_engineering$CREDIT_TO_GOODS_RATIO = feature_engineering$AMT_CREDIT / feature_engineering$AMT_GOODS_PRICE
feature_engineering$INC_PER_CHLD = feature_engineering$AMT_INCOME_TOTAL / (1 + feature_engineering$CNT_CHILDREN)
feature_engineering$SOURCES_PROD = feature_engineering$EXT_SOURCE_1 * feature_engineering$EXT_SOURCE_2 * feature_engineering$EXT_SOURCE_3


#removing rows with more than 70% mssing- compromise between losing too much information and biased dataset.
row_missing_omit <-
  feature_engineering[which(rowMeans(!is.na(feature_engineering)) > 0.7),]



#remove highly correlated features----

numeric_variable <-
  row_missing_omit %>% dplyr::select(where(is.numeric))
#remove problematic variables = FLAG_DOCUMENT_2 , FLAG_MOBIL =var 0 and also doubled
numeric_variable <- numeric_variable %>%
  select(.,-c("FLAG_MOBIL", "FLAG_DOCUMENT_2", LIVINGAREA_MEDI, LIVINGAREA_MODE,ENTRANCES_MEDI, APARTMENTS_MEDI, ELEVATORS_MEDI, LANDAREA_MEDI, BASEMENTAREA_MEDI, NONLIVINGAREA_MEDI))

correlation <-
  corrr::correlate(numeric_variable, method = "pearson")
cor_mat <-
  cor(numeric_variable, use = "complete.obs", method = "pearson")


index = findCorrelation(cor_mat, 0.91)
to_be_removed <- colnames(cor_mat)[index]

#number of removed variables
length(to_be_removed)

#remove  from dataframe
row_missing_omit <- row_missing_omit %>% select(., -(to_be_removed))
clean_data <- row_missing_omit[complete.cases(EXT_SOURCE_3), ]


#value of infinite causes problems hence, it is better to treat as missing
do.call(data.frame, lapply(clean_data, function(x)
  replace(x, is.infinite(x), NA)))
#too small of a category to be removed -unrepresented in the sample
clean_data <-
  clean_data[clean_data$CODE_GENDER != "XNA", , drop = FALSE]
clean_data$CODE_GENDER <- factor(clean_data$CODE_GENDER)

#selecting a random sample
set.seed(42)
clean_data_sample <- clean_data[sample(.N, 100000)]
x <- clean_data_sample[TARGET == 1, .N]
y <- clean_data_sample[TARGET == 0, .N]

class_imbalance = x/y

#missing values imputation median mode
set.seed(42)
clean_data_sample[clean_data_sample == ""] <- NA

clean_data_sample <- clean_data_sample


Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
clean_data_sample <-
  clean_data_sample %>% mutate_if(is.character, as.factor)

clean_data_sample_imputed <-
  clean_data_sample %>% mutate_if(is.numeric, funs(replace(., is.na(.), mean(., na.rm = TRUE)))) %>%
  mutate_if(is.factor, funs(replace(., is.na(.), Mode(na.omit(
    .
  )))))

clean_data_sample_imputed <-
  clean_data_sample_imputed %>% mutate_if(is.character, as.factor)
clean_data_sample_imputed <-
  clean_data_sample_imputed %>% select(TARGET, everything())
write.csv(clean_data_sample_imputed,
          "clean_data_sample_imputed_final.csv")

#Recursive feature elimination

#numeric to factor encoding - required by the method
clean_data_sample_imputed_rfe <- clean_data_sample_imputed %>%
  mutate(TARGET = ifelse(TARGET == 0, "COMPLETE", "DEFAULT"))

clean_data_sample_imputed_rfe <-
  clean_data_sample_imputed_rfe %>% mutate_if(is.character, as.factor)


#setting randomness seed
set.seed(5)
#parallel computing
registerDoParallel(cores = 8)

#long computation time approximately 6hours
rfFuncs$summary <- twoClassSummary

control <-
  rfeControl(
    functions = rfFuncs ,
    method = "repeatedcv",
    number = 3,
    saveDetails = TRUE,
    allowParallel = TRUE
  )

sizes = c(20,40, 50,60,70)
rfeResults = rfe(TARGET ~ .,
                 data = clean_data_sample_imputed_rfe,
                 sizes = sizes,
                 rfeControl = control)


rfeResults$variables

trellis.par.set(caretTheme())
plot(rfeResults, type = c("g", "o"))

#RFE best subset

list_final_rfe <-
  c(
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "SOURCES_PROD"  ,
    "EXT_SOURCE_1"  ,
    "ANNUITY_LENGTH",
    "CREDIT_TO_GOODS_RATIO",
    "LOAN_INCOME_RATIO",
    "INCOME_PER_PERSON",
    "DAYS_BIRTH",
    "NAME_EDUCATION_TYPE",
    "INC_PER_CHLD",
    "AMT_INCOME_TOTAL",
    "DAYS_EMPLOYED",
    "REGION_RATING_CLIENT",
    "AMT_ANNUITY",
    "APARTMENTS_MODE",
    "DAYS_ID_PUBLISH",
    "REGION_RATING_CLIENT_W_CITY",
    "DAYS_REGISTRATION",
    "DEF_60_CNT_SOCIAL_CIRCLE"
  )
length(list_final_rfe)
#information gain selection ---------------------
#removing variables that cannot be handled through this method due to large number of categories.
clean_data_sample_imputed_data <- clean_data_sample_imputed %>% select(.,-TARGET, -ORGANIZATION_TYPE, -OCCUPATION_TYPE)
#one-hot encoding of categorical variables
clean_data_one_hot <- one_hot(clean_data_sample_imputed_data)

clean_data_one_hot$TARGET <- as.factor(clean_data_sample_imputed$TARGET)


weights.ig <- information.gain(TARGET ~., clean_data_one_hot)
#one hot encoding, hence 24vars
subset.ig <- cutoff.k(weights.ig, 24) 
results.ig <- as.simple.formula(subset.ig, "Volume") 


#information best subset
list_info_best <- c(
  "EXT_SOURCE_3",
  "ANNUITY_LENGTH",
  "SOURCES_PROD",
  "EXT_SOURCE_2",
  "EXT_SOURCE_1",
  "DAYS_EMPLOYED",
  "DAYS_BIRTH",
  "CREDIT_TO_GOODS_RATIO",
  "AMT_CREDIT",
  "REGION_RATING_CLIENT_W_CITY",
  "REGION_RATING_CLIENT",
  "NAME_EDUCATION_TYPE",
  "NAME_INCOME_TYPE",
  "DAYS_ID_PUBLISH",
  "DAYS_LAST_PHONE_CHANGE",
  "AMT_ANNUITY",
  "FLOORSMAX_MODE",
  "FLAG_EMP_PHONE",
  "REG_CITY_NOT_WORK_CITY",
  "REGION_POPULATION_RELATIVE"
)
#this clean data set is used in futher code - modelling
write.csv(clean_data_sample_imputed,
          "clean_data_sample_imputed_final.csv")
