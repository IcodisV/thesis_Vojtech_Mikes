# thesis_Vojtech_Mikes
Code to used for my thesis project

Instructions:

Please Download the required dataset from Kaggle - dt.application.train
https://www.kaggle.com/c/home-credit-default-risk

To follow the same preparation technique as done in the thesis, please follow this set structure in running the code:

1. Run Home_credit_data_preparation.R, get clean_data that is write down as csv at the end of the code.

2&3. Run the modelling scripts for Recursive Feature Elimination and Information Gain
-RFE_modelling_home_credit.R
-Information_Gain_modelling_home_credit.R

*each code scripts writes down predictions from each model as csv files

4. Run the final code script which calculates performance results for each of the models
-correct loading of the files is essential
