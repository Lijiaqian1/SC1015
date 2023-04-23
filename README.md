# SC1015 Mini-Project 
We are a team from A124 Group 5. This project is for Introduction to Data Science and Artifical Intelligence module. This project focuses on assisitng doctors, specifically heart diseases.

## Problem Definition
- Assist doctors in dianosis heart disease of a patient
- Assist doctors by suggesting treatment for a patient

## About our Dataset
There were many heart disease data set on Kaggle. But many of them were 'solved' and the datasets were too clean.
We wanted a dataset that is akin to real world data 
The dataset contains 270 case studies 

## Project file Structure
1. Heart.xlsx - Data set from [Kaggle](https://www.kaggle.com/datasets/hephzeebar/heart?resource=download)
2. Disease.ipynb - Main Notebook
3. Webform - Folder containing the webapp for doctor's assistance

## Key takeways from our primary EDA
- There were no strong relationship between heart disease and the features
- Machine learning might be the key to uncover potential nonlinear relationships between the variables
- Reasons why it might be so:
  - Imbalanced dataset
  - Relatively small data size

## Models Used
- Naive Bayes
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting models
  - LightGBM
  - XGBoost
- Multi-Output Regression

## Conclusion
- It is difficult to determine key indicators of heart disease in our dataset as there is no clear relationship between patients' symptoms and presence of heart disease
- The lack of robustness of the dataset was a big challenge in attaining high prediction accuracy, and most of the models implemented faired less than ideally
- However, KNN performed the best in predicting heart disease
- We are able to achieve the secondary objective of predicting patients' treatment as we are able to narrow down a few options suitable for the patient 

## Things we have learnt
- KNN is very good at solving imbalance dataset
- Random forest, SVM, Logistic regression are easily affected by imbalance dataset
- Soft labelling is useful if you do not want your model to be over confident

## Closing Remarks
Our models might not have very high accracy, but we believe that we can apply these models now to be used as a preliminary screening tool to identify patients who are unlikely to have heart disease. This could save time and resources by allowing doctors to focus their attention on patients who are more likely to need further testing.

In the future, when AI can be use in conjunction with doctors, we can greatly reduce the manpower and resouces needed, and allow for more people to access healthcare services.

---

## Contributors
| Name  | Area of Focus | GitHub Acount |
| --- | --- | --- |
| Huang Chien Lun  | Doctor's assitance webapp, managing github/integration  | @alanx0401 |
| Charmaine Low | Data cleaning | @charmainelow1 |
| Li Jiaqian | EDA,machine learning for heart disease & treatment | @Lijiaqian1 |
---


<details>
<summary> References </summary>

- https://flask.palletsprojects.com/en/2.2.x/ 
- https://flask-wtf.readthedocs.io/en/1.0.x/
- https://wtforms.readthedocs.io/en/3.0.x/

</details>


