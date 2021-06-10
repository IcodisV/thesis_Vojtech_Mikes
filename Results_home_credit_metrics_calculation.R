
#required libraries------------
library(ROCR)
library(data.table)
library(readr)
library(MLmetrics)

#RFE no sampling results------------
lgbm_predictions_no_sampling <- read_csv("lgbm_no_sampling_rfe_results_final.csv")
logit_predictions_no_sampling <- read_csv("logit_rfe_no_sampling_results_final.csv")
rf_predictions_no_sampling <- read_csv("results_rf_no_sampling_new_final.csv")
results_xgb_no_sampling <- read_csv("results_xgb_no_sampling_final.csv")
catboost_results_no_sampling_rfe <- read_csv("results_catboost_no_sampling_final.csv")



#catboost

AUC(y_pred = logit_predictions_no_sampling$.pred_1, y_true =logit_predictions_no_sampling$TARGET)
AUC(y_pred = lgbm_predictions_no_sampling$.pred_1, y_true =lgbm_predictions_no_sampling$TARGET)
AUC(y_pred = rf_predictions_no_sampling$.pred_1, y_true =rf_predictions_no_sampling$TARGET)
AUC(y_pred = results_xgb_no_sampling$.pred_1, y_true =results_xgb_no_sampling$TARGET)
AUC(y_pred = catboost_results_no_sampling_rfe$.pred_1, y_true =catboost_results_no_sampling_rfe$TARGET)

pred <- prediction(lgbm_predictions_no_sampling$.pred_1,lgbm_predictions_no_sampling$TARGET)
pred2 <- prediction(logit_predictions_no_sampling$.pred_1, logit_predictions_no_sampling$TARGET)
pred3 <- prediction(rf_predictions_no_sampling$.pred_1,rf_predictions_no_sampling$TARGET)
pred4 <- prediction(results_xgb_no_sampling$.pred_1,results_xgb_no_sampling$TARGET)
pred5 <- prediction(catboost_results_no_sampling_rfe$.pred_1,catboost_results_no_sampling_rfe$TARGET)

perf <- performance( pred, "tpr", "fpr" )
perf2 <- performance(pred2, "tpr", "fpr")
perf3 <- performance(pred3, "tpr", "fpr")
perf4 <- performance(pred4, "tpr", "fpr")
perf5 <- performance(pred5, "tpr", "fpr")
plot(perf2, col = "black")
plot(perf3, add = TRUE, col ="red")
plot(perf, add = TRUE, col = "green")
plot(perf4, add = TRUE, col = "blue")
plot(perf5, add = TRUE, col = "purple")
legend("bottomright", legend = c("Logistic Regression","Random Forest", "LightGBM", "CatBoost", "XGBoost" ),
       col=c(par("fg"), c("red", "green", "blue", "purple")), lwd=2)


#accuracy

Accuracy(catboost_results_no_sampling_rfe$.pred_class, catboost_results_no_sampling_rfe$TARGET)
Accuracy(logit_predictions_no_sampling$.pred_class, logit_predictions_no_sampling$TARGET)
Accuracy(rf_predictions_no_sampling$.pred_class, rf_predictions_no_sampling$TARGET)
Accuracy(lgbm_predictions_no_sampling$.pred_class, lgbm_predictions_no_sampling$TARGET)
Accuracy(results_xgb_no_sampling$.pred_class, results_xgb_no_sampling$TARGET)

#KS

KS_Stat(catboost_results_no_sampling_rfe$.pred_1, catboost_results_no_sampling_rfe$TARGET)
KS_Stat(logit_predictions_no_sampling$.pred_1, logit_predictions_no_sampling$TARGET)
KS_Stat(rf_predictions_no_sampling$.pred_1, rf_predictions_no_sampling$TARGET)
KS_Stat(lgbm_predictions_no_sampling$.pred_1, lgbm_predictions_no_sampling$TARGET)
KS_Stat(results_xgb_no_sampling$.pred_1, results_xgb_no_sampling$TARGET)

#Brier Score

mean((catboost_results_no_sampling_rfe$.pred_1-catboost_results_no_sampling_rfe$TARGET)^2)
mean((logit_predictions_no_sampling$.pred_1-logit_predictions_no_sampling$TARGET)^2)
mean((rf_predictions_no_sampling$.pred_1-rf_predictions_no_sampling$TARGET)^2)
mean((lgbm_predictions_no_sampling$.pred_1-lgbm_predictions_no_sampling$TARGET)^2)
mean((results_xgb_no_sampling$.pred_1-results_xgb_no_sampling$TARGET)^2)


#F1
F1_Score( y_true = catboost_results_no_sampling_rfe$TARGET, y_pred = catboost_results_no_sampling_rfe$.pred_class, positive = NULL)
F1_Score( y_true = logit_predictions_no_sampling$TARGET, y_pred = logit_predictions_no_sampling$.pred_class, positive = NULL)
F1_Score( y_true = rf_predictions_no_sampling$TARGET, y_pred = rf_predictions_no_sampling$.pred_class, positive = NULL)
F1_Score( y_true = lgbm_predictions_no_sampling$TARGET, y_pred = lgbm_predictions_no_sampling$.pred_class, positive = NULL)
F1_Score( y_true = results_xgb_no_sampling$TARGET, y_pred = results_xgb_no_sampling$.pred_class, positive = NULL)

#RFE selection SMOTE------------

lgbm_predictions_smote_sampling <- read_csv("lgbm_smote_sampling_rfe_results_final.csv")
logit_predictions_smote_sampling <- read_csv("logit_rfe_smote_sampling_results_final.csv")
rf_predictions_smote_sampling <-  read_csv("results_rf_smote_sampling_final.csv")
results_xgb_smote_sampling <- read_csv("results_xgb_smote_sampling_final.csv")
results_catboost_smote_sampling <- read_csv("results_catboost_smote_sampling_final.csv")


AUC(y_pred = lgbm_predictions_smote_sampling$.pred_1, y_true =lgbm_predictions_smote_sampling$TARGET)
AUC(y_pred = rf_predictions_smote_sampling$.pred_1, y_true =rf_predictions_smote_sampling$TARGET)
AUC(y_pred = logit_predictions_smote_sampling$.pred_1, y_true =logit_predictions_smote_sampling$TARGET)
AUC(y_pred = results_xgb_smote_sampling$.pred_1, y_true =results_xgb_smote_sampling$TARGET)
AUC(y_pred = results_catboost_smote_sampling$.pred_1, y_true =results_catboost_smote_sampling$TARGET)


pred <- prediction(lgbm_predictions_smote_sampling$.pred_1,lgbm_predictions_smote_sampling$TARGET)
pred2 <- prediction(logit_predictions_smote_sampling$.pred_1, logit_predictions_smote_sampling$TARGET)
pred3 <- prediction(rf_predictions_smote_sampling$.pred_1,rf_predictions_smote_sampling$TARGET)
pred4 <- prediction(results_xgb_smote_sampling$.pred_1,results_xgb_smote_sampling$TARGET)
pred5 <- prediction(results_catboost_smote_sampling$.pred_1,results_catboost_smote_sampling$TARGET)


perf <- performance( pred, "tpr", "fpr" )
perf2 <- performance(pred2, "tpr", "fpr")
perf3 <- performance(pred3, "tpr", "fpr")
perf4 <- performance(pred4, "tpr", "fpr")
perf5 <- performance(pred5, "tpr", "fpr")
plot(perf2, col = "black")
plot(perf3, add = TRUE, col ="red")
plot(perf, add = TRUE, col = "green")
plot(perf4, add = TRUE, col = "blue")
plot(perf5, add = TRUE, col = "purple")
legend("bottomright", legend = c("Logistic Regression","Random Forest", "LightGBM", "CatBoost", "XGBoost" ),
       col=c(par("fg"), c("red", "green", "blue", "purple")), lwd=2)


#accuracy

Accuracy(logit_predictions_smote_sampling$.pred_class, logit_predictions_smote_sampling$TARGET)
Accuracy(rf_predictions_smote_sampling$.pred_class, rf_predictions_smote_sampling$TARGET)
Accuracy(results_xgb_smote_sampling$.pred_class, results_xgb_smote_sampling$TARGET)
Accuracy(lgbm_predictions_smote_sampling$.pred_class, lgbm_predictions_smote_sampling$TARGET)
Accuracy(results_catboost_smote_sampling$.pred_class, results_catboost_smote_sampling$TARGET)


#KS

KS_Stat(results_catboost_smote_sampling$.pred_1, results_catboost_smote_sampling$TARGET)
KS_Stat(logit_predictions_smote_sampling$.pred_1, logit_predictions_smote_sampling$TARGET)
KS_Stat(rf_predictions_smote_sampling$.pred_1, rf_predictions_smote_sampling$TARGET)
KS_Stat(lgbm_predictions_smote_sampling$.pred_1, lgbm_predictions_smote_sampling$TARGET)
KS_Stat(results_xgb_smote_sampling$.pred_1, results_xgb_smote_sampling$TARGET)

#Brier Score

mean((logit_predictions_smote_sampling$.pred_1-logit_predictions_smote_sampling$TARGET)^2)
mean((results_catboost_smote_sampling$.pred_1-results_catboost_smote_sampling$TARGET)^2)
mean((rf_predictions_smote_sampling$.pred_1-rf_predictions_smote_sampling$TARGET)^2)
mean((lgbm_predictions_smote_sampling$.pred_1-lgbm_predictions_smote_sampling$TARGET)^2)
mean((results_xgb_smote_sampling$.pred_1-results_xgb_smote_sampling$TARGET)^2)


#F1
F1_Score( y_true = logit_predictions_smote_sampling$TARGET, y_pred = logit_predictions_smote_sampling$.pred_class, positive = NULL)
F1_Score( y_true = results_catboost_smote_sampling$TARGET, y_pred = results_catboost_smote_sampling$.pred_class, positive = NULL)
F1_Score( y_true = rf_predictions_smote_sampling$TARGET, y_pred = rf_predictions_smote_sampling$.pred_class, positive = NULL)
F1_Score( y_true = lgbm_predictions_smote_sampling$TARGET, y_pred = lgbm_predictions_smote_sampling$.pred_class, positive = NULL)
F1_Score( y_true = results_xgb_smote_sampling$TARGET, y_pred = results_xgb_smote_sampling$.pred_class, positive = NULL)

#Information gain no sampling------------

lgbm_predictions_no_sampling <- read_csv("lgbm_no_sampling_info_results_final.csv")
logit_predictions_no_sampling <- read_csv("logit_info_no_sampling_results_final.csv")
rf_predictions_no_sampling <- read_csv("results_rf_info_no_sampling_final.csv")
results_xgb_no_sampling <- read_csv("results_xgb_info_no_sampling_final.csv")
catboost_results_no_sampling_rfe <- read_csv("results_catboost_info_no_sampling_final.csv")

#catboost

AUC(y_pred = logit_predictions_no_sampling$.pred_1, y_true =logit_predictions_no_sampling$TARGET)
AUC(y_pred = lgbm_predictions_no_sampling$.pred_1, y_true =lgbm_predictions_no_sampling$TARGET)
AUC(y_pred = rf_predictions_no_sampling$.pred_1, y_true =rf_predictions_no_sampling$TARGET)
AUC(y_pred = results_xgb_no_sampling$.pred_1, y_true =results_xgb_no_sampling$TARGET)
AUC(y_pred = catboost_results_no_sampling_rfe$.pred_1, y_true =catboost_results_no_sampling_rfe$TARGET)

pred <- prediction(lgbm_predictions_no_sampling$.pred_1,lgbm_predictions_no_sampling$TARGET)
pred2 <- prediction(logit_predictions_no_sampling$.pred_1, logit_predictions_no_sampling$TARGET)
pred3 <- prediction(rf_predictions_no_sampling$.pred_1,rf_predictions_no_sampling$TARGET)
pred4 <- prediction(results_xgb_no_sampling$.pred_1,results_xgb_no_sampling$TARGET)
pred5 <- prediction(catboost_results_no_sampling_rfe$.pred_1,catboost_results_no_sampling_rfe$TARGET)

perf <- performance( pred, "tpr", "fpr" )
perf2 <- performance(pred2, "tpr", "fpr")
perf3 <- performance(pred3, "tpr", "fpr")
perf4 <- performance(pred4, "tpr", "fpr")
perf5 <- performance(pred5, "tpr", "fpr")

plot(perf, , col = "green")
plot(perf2,add = TRUE, col = "black")
plot(perf3, add = TRUE, col ="red")
plot(perf4, add = TRUE, col = "blue")
plot(perf5, add = TRUE, col = "purple")
legend("bottomright", legend = c("Logistic Regression","Random Forest", "LightGBM",  "XGBoost", "CatBoost"),
       col=c(par("fg"), c("red", "green", "blue", "purple")), lwd=2)


#accuracy

Accuracy(catboost_results_no_sampling_rfe$.pred_class, catboost_results_no_sampling_rfe$TARGET)
Accuracy(logit_predictions_no_sampling$.pred_class, logit_predictions_no_sampling$TARGET)
Accuracy(rf_predictions_no_sampling$.pred_class, rf_predictions_no_sampling$TARGET)
Accuracy(lgbm_predictions_no_sampling$.pred_class, lgbm_predictions_no_sampling$TARGET)
Accuracy(results_xgb_no_sampling$.pred_class, results_xgb_no_sampling$TARGET)

#KS

KS_Stat(catboost_results_no_sampling_rfe$.pred_1, catboost_results_no_sampling_rfe$TARGET)
KS_Stat(logit_predictions_no_sampling$.pred_1, logit_predictions_no_sampling$TARGET)
KS_Stat(rf_predictions_no_sampling$.pred_1, rf_predictions_no_sampling$TARGET)
KS_Stat(lgbm_predictions_no_sampling$.pred_1, lgbm_predictions_no_sampling$TARGET)
KS_Stat(results_xgb_no_sampling$.pred_1, results_xgb_no_sampling$TARGET)

#Brier Score

mean((catboost_results_no_sampling_rfe$.pred_1-catboost_results_no_sampling_rfe$TARGET)^2)
mean((logit_predictions_no_sampling$.pred_1-logit_predictions_no_sampling$TARGET)^2)
mean((rf_predictions_no_sampling$.pred_1-rf_predictions_no_sampling$TARGET)^2)
mean((lgbm_predictions_no_sampling$.pred_1-lgbm_predictions_no_sampling$TARGET)^2)
mean((results_xgb_no_sampling$.pred_1-results_xgb_no_sampling$TARGET)^2)


#F1
F1_Score( y_true = catboost_results_no_sampling_rfe$TARGET, y_pred = catboost_results_no_sampling_rfe$.pred_class, positive = NULL)
F1_Score( y_true = logit_predictions_no_sampling$TARGET, y_pred = logit_predictions_no_sampling$.pred_class, positive = NULL)
F1_Score( y_true = rf_predictions_no_sampling$TARGET, y_pred = rf_predictions_no_sampling$.pred_class, positive = NULL)
F1_Score( y_true = lgbm_predictions_no_sampling$TARGET, y_pred = lgbm_predictions_no_sampling$.pred_class, positive = NULL)
F1_Score( y_true = results_xgb_no_sampling$TARGET, y_pred = results_xgb_no_sampling$.pred_class, positive = NULL)



#information gain smote------------

lgbm_predictions_smote_sampling <- read_csv("lgbm_smote_sampling_info_results_final.csv")
logit_predictions_smote_sampling <- read_csv("logit_info_smote_sampling_results_final.csv")
rf_predictions_smote_sampling <-  read_csv("results_rf_info_smote_sampling_final.csv")
results_xgb_smote_sampling <- read_csv("results_xgb_info_smote_sampling_final.csv")
results_catboost_smote_sampling <- read_csv("results_catboost_info_smote_sampling_final.csv")

#catboost
#lgbm

AUC(y_pred = lgbm_predictions_smote_sampling$.pred_1, y_true =lgbm_predictions_smote_sampling$TARGET)
AUC(y_pred = rf_predictions_smote_sampling$.pred_1, y_true =rf_predictions_smote_sampling$TARGET)
AUC(y_pred = logit_predictions_smote_sampling$.pred_1, y_true =logit_predictions_smote_sampling$TARGET)
AUC(y_pred = results_xgb_smote_sampling$.pred_1, y_true =results_xgb_smote_sampling$TARGET)
AUC(y_pred = results_catboost_smote_sampling$.pred_1, y_true =results_catboost_smote_sampling$TARGET)


pred <- prediction(lgbm_predictions_smote_sampling$.pred_1,lgbm_predictions_smote_sampling$TARGET)
pred2 <- prediction(logit_predictions_smote_sampling$.pred_1, logit_predictions_smote_sampling$TARGET)
pred3 <- prediction(rf_predictions_smote_sampling$.pred_1,rf_predictions_smote_sampling$TARGET)
pred4 <- prediction(results_xgb_smote_sampling$.pred_1,results_xgb_smote_sampling$TARGET)
pred5 <- prediction(results_catboost_smote_sampling$.pred_1,results_catboost_smote_sampling$TARGET)


perf <- performance( pred, "tpr", "fpr" )
perf2 <- performance(pred2, "tpr", "fpr")
perf3 <- performance(pred3, "tpr", "fpr")
perf4 <- performance(pred4, "tpr", "fpr")
perf5 <- performance(pred5, "tpr", "fpr")
plot(perf2, col = "black")
plot(perf3, add = TRUE, col ="red")
plot(perf, add = TRUE, col = "green")
plot(perf4, add = TRUE, col = "blue")
plot(perf5, add = TRUE, col = "purple")
legend("bottomright", legend = c("Logistic Regression","Random Forest", "LightGBM", "CatBoost", "XGBoost" ),
       col=c(par("fg"), c("red", "green", "blue", "purple")), lwd=2)


#accuracy

Accuracy(logit_predictions_smote_sampling$.pred_class, logit_predictions_smote_sampling$TARGET)
Accuracy(rf_predictions_smote_sampling$.pred_class, rf_predictions_smote_sampling$TARGET)
Accuracy(results_xgb_smote_sampling$.pred_class, results_xgb_smote_sampling$TARGET)
Accuracy(lgbm_predictions_smote_sampling$.pred_class, lgbm_predictions_smote_sampling$TARGET)
Accuracy(results_catboost_smote_sampling$.pred_class, results_catboost_smote_sampling$TARGET)


#KS

KS_Stat(results_catboost_smote_sampling$.pred_1, results_catboost_smote_sampling$TARGET)
KS_Stat(logit_predictions_smote_sampling$.pred_1, logit_predictions_smote_sampling$TARGET)
KS_Stat(rf_predictions_smote_sampling$.pred_1, rf_predictions_smote_sampling$TARGET)
KS_Stat(lgbm_predictions_smote_sampling$.pred_1, lgbm_predictions_smote_sampling$TARGET)
KS_Stat(results_xgb_smote_sampling$.pred_1, results_xgb_smote_sampling$TARGET)

#Brier Score

mean((logit_predictions_smote_sampling$.pred_1-logit_predictions_smote_sampling$TARGET)^2)
mean((results_catboost_smote_sampling$.pred_1-results_catboost_smote_sampling$TARGET)^2)
mean((rf_predictions_smote_sampling$.pred_1-rf_predictions_smote_sampling$TARGET)^2)
mean((lgbm_predictions_smote_sampling$.pred_1-lgbm_predictions_smote_sampling$TARGET)^2)
mean((results_xgb_smote_sampling$.pred_1-results_xgb_smote_sampling$TARGET)^2)


#F1
F1_Score( y_true = logit_predictions_smote_sampling$TARGET, y_pred = logit_predictions_smote_sampling$.pred_class, positive = NULL)
F1_Score( y_true = results_catboost_smote_sampling$TARGET, y_pred = results_catboost_smote_sampling$.pred_class, positive = NULL)
F1_Score( y_true = rf_predictions_smote_sampling$TARGET, y_pred = rf_predictions_smote_sampling$.pred_class, positive = NULL)
F1_Score( y_true = lgbm_predictions_smote_sampling$TARGET, y_pred = lgbm_predictions_smote_sampling$.pred_class, positive = NULL)
F1_Score( y_true = results_xgb_smote_sampling$TARGET, y_pred = results_xgb_smote_sampling$.pred_class, positive = NULL)