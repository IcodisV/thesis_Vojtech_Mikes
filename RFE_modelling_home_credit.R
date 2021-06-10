
#required libraries----------------
library(readr)
library(data.table)
library(tidymodels)
library(skimr)
library(corrr)
library(knitr)
library(MLmetrics)
library("themis")
library("ranger")
library("doParallel")
library("vip")
library("ggridges")
library("xgboost")
library(yardstick)
library(onehot)
library(mltools)
library(treesnip)
library(lightgbm) #requires custom installation - difficult
library(catboost) #requires custom installation - difficult
library(SHAPforxgboost)
library(lime)
library(magrittr)


#loading the data prepared in the preparation script----
data <- read.csv("clean_home_credit_data.csv")


set.seed(42)
#selecting RFE subset as determined in the preparation part.
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

data_rfe <- data %>% select(TARGET, list_final_rfe)

#character variables to factor
data_rfe <- data_rfe%>%mutate_if(is.character, as.factor)
data_rfe$TARGET <- as.factor(data_rfe$TARGET)
data_rfe <- as.data.table(data_rfe)


#test/train RFE split----

#= numeric 75% training 25% testing, stratified sampling on response class

split_selection <- initial_split(data_rfe, prob = 3/4, strata = TARGET)
imbalance_training <- training(split_selection)
imbalance_test <- testing(split_selection)


#logistic regression RFE - no sampling ------------
#preproccesing of the recipe
logreg_recipe  <-
  recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(),-all_outcomes())%>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_corr(all_numeric(), -all_outcomes(), threshold = 0.9) %>%
  step_other(all_nominal(),-all_outcomes())%>%
  step_zv(all_predictors(), -all_outcomes())
  prep()


#setting up the engine
logreg_model <- logistic_reg() %>% 
  set_engine("glm", control = list(maxit = 50))%>%
  set_mode("classification") %>%
  translate()


logreg_wf <- workflow() %>% 
  add_recipe(logreg_recipe) %>% 
  add_model(logreg_model)

#training on training data
logreg_fit <- logreg_wf %>% fit(imbalance_training)

tidy(logreg_fit) %>% arrange(desc(std.error))

#fitting on test set - binary
logreg_pred <- logreg_fit %>% predict(new_data = imbalance_test)

#collecting metrics
class_metrics <- metric_set(accuracy, f_meas, kap, specificity, sensitivity)
#fitting on test set - probability prediction
predictions <- logreg_fit %>%
  predict(new_data = imbalance_test, type = "prob")


predictions_df <- imbalance_test %>% bind_cols(logreg_pred) %>% bind_cols(predictions)

logreg_test_metrics1 <- 
  imbalance_test %>% bind_cols(logreg_pred) %>% bind_cols(predictions) %>%
  class_metrics(truth = TARGET, estimate = .pred_class)

logit_rfe_no_sampling_results <- predictions_df %>% select(TARGET, .pred_class, .pred_1)

write.csv(logit_rfe_no_sampling_results, "logit_rfe_no_sampling_results_final.csv")

#logistic regression RFE SMOTE sampling--------------
#preproccesing of the recipe - SMOTE sampling added 20% balancing
logreg_recipe_smote  <-
  recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(),-all_outcomes())%>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_corr(all_numeric(), -all_outcomes(), threshold = 0.9) %>%
  step_zv(all_predictors()) %>%
  step_naomit(all_predictors()) %>%
  step_smote(TARGET, over_ratio = 0.2)

#setting up the engine
logreg_model_smote <- logistic_reg() %>% 
  set_engine("glm", control = list(maxit = 50))%>%
  set_mode("classification") %>%
  translate()


logreg_wf_smote <- workflow() %>% 
  add_recipe(logreg_recipe_smote) %>% 
  add_model(logreg_model_smote)

#training on training data
logreg_fit_smote <- logreg_wf_smote %>% fit(imbalance_training)

tidy(logreg_fit_smote) %>% arrange(desc(std.error))

#fitting on test set - binary
logreg_pred_smote <- logreg_fit_smote %>% predict(new_data = imbalance_test)

class_metrics <- metric_set(accuracy, f_meas, kap, specificity, sensitivity)

#fitting on test set- probability prediction
predictions_smote <- logreg_fit_smote %>%
  predict(new_data = imbalance_test, type = "prob")


predictions_df_smote <- imbalance_test %>% bind_cols(logreg_pred_smote) %>% bind_cols(predictions_smote)
logreg_test_metrics_smote <- predictions_df_smote %>%class_metrics(truth = TARGET, estimate = .pred_class)

logit_rfe_smote_results <- predictions_df_smote %>% select(TARGET, .pred_class, .pred_1)

write.csv(logit_rfe_smote_results, "logit_rfe_smote_sampling_results_final.csv")


#random forest RFE no sampling------------------

set.seed(3)

# cross validation using k folds (k = 5), stratified sampling
cv_folds <-  imbalance_training%>% vfold_cv(v = 5, strata = TARGET)

# recipe pre-processing
rfe_balanced_recipe <-  recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(),-all_outcomes())%>%
  step_nzv(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_predictors())

rfe_balanced_recipe

# setting up engine
rf_model_tune <- rand_forest(mtry = tune(), trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger")

rf_tune_wf <- workflow() %>%
  add_recipe(rfe_balanced_recipe) %>%
  add_model(rf_model_tune)
rf_tune_wf

class_and_probs_metrics <- metric_set(accuracy, kap, sensitivity, 
                                      specificity, roc_auc)

set.seed(99154345)

#setting up tuning grid for mtry parameter of random forest - Number of variables randomly sampled as candidates at each split
rf_grid <- expand.grid(
  mtry = c(5:15)
)

registerDoParallel(cores = 8)

#tuning of the model
rf_tune_res <- tune_grid(
  rf_tune_wf,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = class_and_probs_metrics
)

rf_tune_res %>%
  collect_metrics()

#select best model based on highest ROC_AUC

best_acc <- select_best(rf_tune_res, metric="roc_auc")
rf_final_wf <- finalize_workflow(rf_tune_wf, best_acc)
rf_final_wf

#fitting on the test set
rf_final_fit <- rf_final_wf %>%
  last_fit(split_selection, metrics = class_and_probs_metrics)

#collecting metrics and predictions - saving the results
rf_final_fit %>%
  collect_metrics()
results <- rf_final_fit %>% collect_predictions()
results_rf_no_sampling <- results%>%  select(TARGET, .pred_1, .pred_class)
write.csv(results_rf_no_sampling, "results_rf_no_sampling_new_final.csv")

#random forest  RFE smote sampling-----------------


#recipe pre-processing - including SMOTE oversampling the minority group to 20%
rfe_balanced_recipe <-  recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(),-all_outcomes())%>%
  step_nzv(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_predictors())%>%
  step_smote(TARGET, over_ratio = 0.2)

#setting up the engine
rf_model_tune <- rand_forest(mtry = tune(), trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger")

rf_tune_wf <- workflow() %>%
  add_recipe(rfe_balanced_recipe) %>%
  add_model(rf_model_tune)
rf_tune_wf$
  
class_and_probs_metrics <- metric_set(accuracy, f_meas, sensitivity, specificity, roc_auc)

#setting up the tuning grid
rf_grid <- expand.grid(
  mtry = c(5:15)
)

registerDoParallel(cores = 8)

#conducting tuning
rf_tune_res <- tune_grid(
  rf_tune_wf,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = class_and_probs_metrics
)
rf_tune_res %>%
  collect_metrics()


#selecting best tuning parameters based on ROC-AUC
best_acc <- select_best(rf_tune_res, "roc_auc")
rf_final_wf <- finalize_workflow(rf_tune_wf, best_acc)
rf_final_wf

set.seed(99154345)

#fitting on the testing set
rf_final_fit <- rf_final_wf %>%
  last_fit(split_selection, metrics = class_and_probs_metrics)

#collecting metrics and predictions - saving the results.
rf_final_fit %>%
  collect_metrics(class_and_probs_metrics)

results <- rf_final_fit %>% collect_predictions()

results_rf_smote_sampling <- results%>%  select(TARGET, .pred_1, .pred_class)

write.csv(results_rf_smote_sampling, "results_rf_smote_sampling_final.csv")

#XGboost RFE no sampling-------------

# recipe pre-processing - one-hot encoding of categorical variables
xgb_recipe <- recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(),-all_outcomes(), one_hot = TRUE)%>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) 

# setting up the engine
xgb_model_tune <- 
  boost_tree(trees = tune(), tree_depth = tune(), 
             learn_rate = tune(), stop_iter = 500) %>%
  set_mode("classification") %>%
  set_engine("xgboost")


xgb_tune_wf <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_model_tune)
xgb_tune_wf

class_and_probs_metrics <- metric_set(accuracy, kap, sensitivity, 
                                      specificity, roc_auc)

#setting up tuning grid for mtry parameter of XGBOOST - Learn Rate - importance given to misclassified cases
# tree_depth - Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
set.seed(8504)
grid_max_entropy(trees(range = c(0, 500)), 
                 learn_rate(range = c(-2, -1)), 
                 tree_depth(), size = 20)

xgb_grid <- expand.grid(trees = 500 *3, 
                        learn_rate = c(0.01:0.1), 
                        tree_depth = 3:10)

registerDoParallel(cores = 6)

#conducting tuning
xgb_tune_res <- tune_grid(
  xgb_tune_wf,
  resamples = cv_folds,
  grid = xgb_grid,
  metrics = class_and_probs_metrics
)

xgb_tune_metrics <- xgb_tune_res %>%
  collect_metrics()
xgb_tune_metrics

#selecting best based on ROC_AUC
xgb_best <- xgb_tune_metrics %>% 
  filter(.metric == "roc_auc", tree_depth == 3, learn_rate == 0.01, trees == 1500)
xgb_final_wf <- finalize_workflow(xgb_tune_wf, xgb_best)
xgb_final_wf

#fitting on the test set
xgb_final_fit <- xgb_final_wf %>%
  last_fit(split_selection, metrics = class_and_probs_metrics)

#collecting metrics and predicitons - saving results.
xgb_final_fit %>%
  collect_metrics()

results <- xgb_final_fit %>% collect_predictions()

results_xgb_no_sampling <- results%>%  select(TARGET, .pred_1, .pred_class)

write.csv(results_xgb_no_sampling, "results_xgb_no_sampling_final.csv")



#XGboost RFE SMOTE sampling -----

# recipe pre-processing - one-hot encoding of categorical variables
#including smote oversampling
xgb_recipe <- recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(),-all_outcomes(), one_hot = TRUE)%>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_smote(TARGET, over_ratio = 0.2)

#setting up the engine
xgb_model_tune <- 
  boost_tree(trees = tune(), tree_depth = tune(), 
             learn_rate = tune(), stop_iter = 500) %>%
  set_mode("classification") %>%
  set_engine("xgboost")


xgb_tune_wf <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_model_tune)
xgb_tune_wf

class_and_probs_metrics <- metric_set(accuracy, kap, sensitivity, 
                                      specificity, roc_auc)

set.seed(8504)

#setting up tuning grid for mtry parameter of XGBOOST - Learn Rate - importance given to misclassified cases
# tree_depth - Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
grid_max_entropy(trees(range = c(0, 500)), 
                 learn_rate(range = c(-2, -1)), 
                 tree_depth(), size = 20)

xgb_grid <- expand.grid(trees = 500 * 1:3, 
                        learn_rate = c(0.01:0.1), 
                        tree_depth = c(3:10))
registerDoParallel(cores = 8)

#conducting tuning
xgb_tune_res <- tune_grid(
  xgb_tune_wf,
  resamples = cv_folds,
  grid = xgb_grid,
  metrics = class_and_probs_metrics,
  control = control_grid(verbose = TRUE)
)

xgb_tune_metrics <- xgb_tune_res %>%
  collect_metrics()
xgb_tune_metrics

#selecting best based on ROC_AUC
xgb_best <- xgb_tune_metrics %>% 
  filter(.metric == "roc_auc", tree_depth == 3, learn_rate == 0.01, trees == 1500)
xgb_final_wf <- finalize_workflow(xgb_tune_wf, xgb_best)
xgb_final_wf

#fitting on the test set
xgb_final_fit <- xgb_final_wf %>%
  last_fit(split_selection, metrics = class_and_probs_metrics)

#collecting metrics and predicitons - saving the results.
xgb_final_fit %>%
  collect_metrics()

results <- xgb_final_fit %>% collect_predictions()
results_xgb_smote_sampling <- results%>%  select(TARGET, .pred_1, .pred_class)
write.csv(results_xgb_smote_sampling, "results_xgb_smote_sampling_final.csv")

#catboost RFE no sampling----------


# recipe pre-processing - one-hot encoding of categorical variables
recipe_spec <- recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)%>%
  step_zv(all_predictors())%>%
  step_impute_mean(all_predictors())


# setting up the engine
model_spec_catboost_tune <- boost_tree(
  mode           = "classification",
  trees          = 1000,
  mtry            = tune(),
  min_n          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune()
  #            loss_reduction = tune(),
) %>%
  set_engine("catboost",  nthread = 8)


#setting up tuning grid for mtry parameter of CatBoost - Learn Rate - importance given to misclassified cases
# tree_depth - Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
#min_n = The minimum number of data points in a node that are required for the node to be split further.
catboost_params <-
  dials::parameters(
    finalize(mtry(), imbalance_training),
    min_n(),
    tree_depth(range = c(5, 1)),
    learn_rate(range = c(-0.1, -1), trans = log10_trans())
  )

catboost_grid <-
  dials::grid_max_entropy(catboost_params,
                          size =20)

wflw_spec_catboost_tune <- workflow() %>%
  add_model(model_spec_catboost_tune) %>%
  add_recipe(recipe_spec)

set_dependency("boost_tree", eng = "catboost", "catboost")
set_dependency("boost_tree", eng = "catboost", "treesnip")


#conducting tuning
tune_results_catboost <- tune_grid(
  object     = wflw_spec_catboost_tune,
  resamples  = cv_folds,
  param_info = parameters(wflw_spec_catboost_tune),
  grid       = catboost_grid,
  control    = control_grid(verbose = TRUE, allow_par = TRUE)
)

catboost_tune_metrics <- tune_results_catboost %>% collect_metrics()

best_results_catboost <- tune_results_catboost %>%
  show_best(metric = "roc_auc", n = 30)

#selecting best based on ROC_AUC
best_acc <- select_best(tune_results_catboost, "roc_auc")
catboost_final_wf <- finalize_workflow(wflw_spec_catboost_tune, best_acc)
catboost_final_wf

#fitting on the test set
catboost_final_fit <- catboost_final_wf %>%
  last_fit(split_selection)

#collecting metrics and predicitons - saving results.
catboost_final_fit %>%
  collect_metrics()
catboost_final_fit %>%
  collect_predictions()
results_catboost_no_sampling <- catboost_final_fit%>% collect_predictions() %>% select(TARGET, .pred_1, .pred_class)
write.csv(results_catboost_no_sampling, "results_catboost_no_sampling_final.csv")

##catboost RFE SMOTE sampling --------------

# recipe pre-processing - one-hot encoding of categorical variables
#including smote oversampling
recipe_spec <- recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)%>%
  step_zv(all_predictors())%>%
  step_impute_mean(all_predictors())%>%
  step_smote(TARGET, over_ratio = 0.2)

#setting up the engine
model_spec_catboost_tune <- boost_tree(
  mode           = "classification",
  trees          = 1000,
  mtry            = tune(),
  min_n          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune()
  #            loss_reduction = tune(),
) %>%
  set_engine("catboost",  nthread = 8)

#setting up tuning grid for mtry parameter of CatBoost - Learn Rate - importance given to misclassified cases
# tree_depth - Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
#min_n = The minimum number of data points in a node that are required for the node to be split further.
catboost_params <-
  dials::parameters(
    finalize(mtry(), imbalance_training),
    min_n(),
    tree_depth(range = c(3,  1)),
    learn_rate(range = c(-0.1, -1), trans = log10_trans())
  )

catboost_grid <-
  dials::grid_max_entropy(catboost_params,
                          size =20)

wflw_spec_catboost_tune <- workflow() %>%
  add_model(model_spec_catboost_tune) %>%
  add_recipe(recipe_spec)

set_dependency("boost_tree", eng = "catboost", "catboost")
set_dependency("boost_tree", eng = "catboost", "treesnip")

#conducting tuning
tune_results_catboost <- tune_grid(
  object     = wflw_spec_catboost_tune,
  resamples  = cv_folds,
  param_info = parameters(wflw_spec_catboost_tune),
  grid       = catboost_grid,
  control    = control_grid(verbose = TRUE, allow_par = TRUE)
)

catboost_tune_metrics <- tune_results_catboost %>% collect_metrics()

best_results_catboost <- tune_results_catboost %>%
  show_best(metric = "roc_auc", n = 30)

#selecting best tuning parameters based on ROC_AUC
best_acc <- select_best(tune_results_catboost, "roc_auc")
catboost_final_wf <- finalize_workflow(wflw_spec_catboost_tune, best_acc)
catboost_final_wf

#fitting on the test set
catboost_final_fit <- catboost_final_wf %>%
  last_fit(split_selection)


#collecting metrics and predictions - saving results.
catboost_final_fit_results_smote <- catboost_final_fit %>%
  collect_predictions()

catboost_final_fit %>%
  collect_metrics()

results_catboost_smote_sampling <- catboost_final_fit_results_smote%>%  select(TARGET, .pred_1, .pred_class)

write.csv(results_catboost_smote_sampling, "results_catboost_smote_sampling_final.csv")


#lightGBM RFE- no sampling including lime ---------------


# recipe pre-processing - one-hot encoding of categorical variables
recipe_spec <- recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(),-all_outcomes(), one_hot = TRUE)%>%
  step_zv(all_predictors())%>%
  step_impute_mean(all_predictors())

# setting up the engine
model_spec_lightGBM_tune <- boost_tree(
  mode           = "classification",
  trees          = 1000,
  mtry            = tune(),
  min_n          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune()
) %>%
  set_engine("lightgbm",  nthread = 4)


#setting up tuning grid for mtry parameter of lightGBM - Learn Rate - importance given to misclassified cases
# tree_depth - Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
#min_n = The minimum number of data points in a node that are required for the node to be split further.
# loss_reduction = Minimum loss reduction required to make a further partition on a leaf node of the tree.
lightGBM_params <-
  dials::parameters(
    finalize(mtry(), imbalance_training),
    min_n(),
    tree_depth(range = c(5, 1)),
    learn_rate(range = c(-0.01, -0.5), trans = log10_trans()),
    loss_reduction()
  )


lightGBM_grid <-
  dials::grid_max_entropy(lightGBM_params,
                          size = 20)

wflw_spec_lightGBM_tune <- workflow() %>%
  add_model(model_spec_lightGBM_tune) %>%
  add_recipe(recipe_spec)

set_dependency("boost_tree", eng = "lightgbm", "lightgbm")
set_dependency("boost_tree", eng = "lightgbm", "treesnip")
#conducting tuning
tune_results_lightGBM <- tune_grid(
  object     = wflw_spec_lightGBM_tune,
  resamples  = cv_folds,
  param_info = parameters(wflw_spec_lightGBM_tune),
  grid       = lightGBM_grid,
  control    = control_grid(verbose = TRUE, allow_par = TRUE)
)

lightGBM_tune_metrics <- tune_results_lightGBM %>% collect_metrics()

best_results_lightGBM <- tune_results_lightGBM %>%
  show_best(metric = "roc_auc", n = 30)


#selecting best tuning parameters based on ROC_AUC
best_acc <- select_best(tune_results_lightGBM, "roc_auc")
lightGBM_final_wf <- finalize_workflow(wflw_spec_lightGBM_tune, best_acc)
lightGBM_final_wf

#fitting on the test set
lightGBM_final_fit <- lightGBM_final_wf %>%
  last_fit(split_selection)


#collecting metrics and predicitons - saving results.
lightGBM_final_fit %>%
  collect_metrics()

lgbm_no_sampling_rfe_results <- lightGBM_final_fit %>% collect_predictions()
lgbm_no_sampling_rfe_results <- lgbm_no_sampling_rfe_results %>% select(.pred_0, .pred_1, .pred_class, TARGET)
write.csv(lgbm_no_sampling_rfe_results, "lgbm_no_sampling_rfe_results.csv")

#lime analysis

model_spec_lightgbm_tune_lime <- boost_tree(
  mode           = "classification",
  trees          = 1000,
  mtry            = 14,
  min_n          = 4,
  tree_depth     = 1,
  learn_rate     = 0.710752713518268,
  loss_reduction = 1.01882917323322e-06
) 

model_spec_lightgbm_tune_lime <-  set_engine(model_spec_lightgbm_tune_lime, engine = "lightgbm",  nthread = 4)
training_data <- imbalance_training %>% select(., -TARGET)
model <- fit_xy(object = model_spec_lightgbm_tune_lime, x = select(imbalance_training, -TARGET), y = select(imbalance_training, TARGET))
training_data <- imbalance_training %>% select(., -TARGET)
#random sample of two observations
training_data <- training_data[sample(.N, 2)]
explainer <- lime(x = imbalance_training %>% select(., -TARGET), model = model)

explanation <- explain(x = training_data, labels = "1",
                       n_permutations = 500,
                       dist_fun = "manhattan",
                       explainer = explainer,
                       kernel_width = 3,
                       n_features = 10)

plot_features(explanation)


#lightGBM RFE - SMOTE sampling-------------

#RFE lightGBM with smote
recipe_spec <- recipe(TARGET ~ ., data = imbalance_training) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)%>%
  step_zv(all_predictors())%>%
  step_smote(TARGET, over_ratio = 0.2) 

model_spec_lightGBM_tune <- boost_tree(
  mode           = "classification",
  trees          = 1000,
  mtry            = tune(),
  min_n          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune()
) %>%
  set_engine("lightgbm",  nthread = 8)

lightgbm_params <- 
  dials::parameters(
    min_n(),
    tree_depth(range = c(3, 8)),
    learn_rate(range = c(-3, -1), trans = log10_trans()),
    loss_reduction(),
    finalize(mtry(), imbalance_training))

lightGBM_grid <-
  dials::grid_max_entropy(lightgbm_params,
                          size =20)

wflw_spec_lightGBM_tune <- workflow() %>%
  add_model(model_spec_lightGBM_tune) %>%
  add_recipe(recipe_spec)

set_dependency("boost_tree", eng = "lightgbm", "lightgbm")
set_dependency("boost_tree", eng = "lightgbm", "treesnip")

tune_results_lightGBM <- tune_grid(
  object     = wflw_spec_lightGBM_tune,
  resamples  = cv_folds,
  param_info = parameters(wflw_spec_lightGBM_tune),
  grid       = lightGBM_grid,
  control    = control_grid(verbose = TRUE, allow_par = TRUE)
)

lightGBM_tune_metrics <- tune_results_lightGBM %>% collect_metrics()

best_results_lightGBM <- tune_results_lightGBM %>%
  show_best(metric = "roc_auc", n = 30)

best_acc <- select_best(tune_results_lightGBM, "roc_auc")
lightGBM_final_wf <- finalize_workflow(wflw_spec_lightGBM_tune, best_acc)
lightGBM_final_wf


lightGBM_final_fit <- lightGBM_final_wf %>%
  last_fit(split_selection)

lightGBM_final_fit %>%
  collect_metrics()

lgbm_smote_sampling_rfe_results <- lightGBM_final_fit %>% collect_predictions()
lgbm_smote_sampling_rfe_results <- lgbm_smote_sampling_rfe_results %>% select(.pred_0, .pred_1, .pred_class, TARGET)

write.csv(lgbm_smote_sampling_rfe_results, "lgbm_smote_sampling_rfe_results.csv")

#shap analysis different lightGBM framework required -----------------
#requires  external one hot encoding - 
data <- data%>%mutate_if(is.character, as.factor)
data$TARGET <- as.factor(data$TARGET)
data <- as.data.table(data)
data_one_hot <- one_hot(data[,-1])
data_one_hot$TARGET <-data$TARGET
data_one_hot <- Filter(function(x)(length(unique(x))>1), data_one_hot)

set.seed(42)
training -validation -test splot
split_selection <- initial_split(data_one_hot, prob = 3/4, strata = TARGET)
training_lgbm <- training(split_selection)

#the response needs to be numeric
training_lgbm <- training_lgbm %>%
  mutate(TARGET = ifelse(TARGET == "0",0,1))
testing_lgbm <- testing(split_selection)

testing_lgbm <- testing_lgbm %>%
  mutate(TARGET = ifelse(TARGET == "0",0,1))

# Define a new partition to split the remaining 25%
tune_plus_val_index <- createDataPartition(testing_lgbm$TARGET,
                                           p = .9,
                                           list = FALSE,
                                           times = 1)

# Split the remaining ~25% of the data: 40% (tune) and 60% (val)
tune_data <- testing_lgbm[-tune_plus_val_index, ,drop=FALSE]
val_data <- testing_lgbm[tune_plus_val_index, ,drop=FALSE]

train_data_matrix <- training_lgbm %>% select(.,-TARGET)

#formatting the data in the required format by lightGBM
bank_rules <- lgb.convert_with_rules(data = train_data_matrix)
bank_train <- bank_rules$data

#separating response var from training data
train_data_matrix <- data.matrix(bank_train)

train_label = training_lgbm$TARGET

#separating response var from validation data
tune_data_data <- tune_data %>% select(.,-TARGET)
bank_tune <- lgb.convert_with_rules(data = tune_data_data, rules = bank_rules$rules)$data

bank_val <- lgb.convert_with_rules(data = val_data, rules = bank_rules$rules)$data
tune_data_matrix <- data.matrix(bank_tune)
tune_label = tune_data$TARGET

#separating response var from testing data
dtest<- data.matrix(bank_val%>% select(.,-TARGET))
dtest_label <- bank_val$TARGET

#prepared data for lightGBM
lgb.train <- lgb.Dataset(train_data_matrix, label = train_label)
lgb.valid <- lgb.Dataset(tune_data_matrix, label = tune_label)

#best tuning parameters - this original framework includes more tuning parameters 
lgb.params<- list(objective = "binary",
                  metric = "auc",
                  num_leaves = 32,
                  max_depth=10	,
                  num_threads = 2,
                  min_data_in_leaf = 14	,
                  min_sum_hessian_in_leaf = 0.05,
                  feature_fraction = 0.8,
                  bagging_fraction = 0.6,
                  bagging_freq = 2,
                  lambda_l1 = 0.4, 
                  lambda_l2 = 0.4,
                  min_gain_to_split=0.2
)

#training of the model
lgb.model <- lgb.train(params = lgb.params,
                       data = lgb.train,
                       valids = list(val = lgb.valid),
                       learning_rate = 0.02,
                       nrounds = 5000,
                       early_stopping_rounds = 200,
                       eval_freq = 50
)


#fitting on the test set
predictions <-  lgb.model%>% predict( data = dtest, n = lgb.model$best_iter)


#calculating SHAP values
shap_values <- shap.values(xgb_model = lgb.model, X_train = data.matrix(training_lgbm %>% select(.,-TARGET)))

#formatting shap values in the long format
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train =  data.matrix(training_lgbm %>% select(.,-TARGET)), top_n = 10)

shap.importance(shap_long, names_only = FALSE, top_n = 10)

#plotting summary of SHAP importance
summary <- shap.plot.summary(shap_long, dilute = 5)

#dependence plots SHAP
fig_list <- lapply(names(shap_values$mean_shap_score)[1:6], 
                   shap.plot.dependence, data_long = shap_long)
gridExtra::grid.arrange(grobs = fig_list, ncol = 2)