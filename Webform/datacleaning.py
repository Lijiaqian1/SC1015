import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from ydata_profiling import ProfileReport
#Profile report takes very long to generate change the bool below to allow for report generation
togeneratereport = False
sb.set() # set the default Seaborn style for graphics

heartdata = pd.read_excel("Heart.xlsx")
if togeneratereport:
    profile = ProfileReport(heartdata, title="Profiling Report Before Data Cleaning")
    profile.to_notebook_iframe()
heartdata = heartdata.drop(columns=['Name'])

heartdata.info()

#create columns for each treatment 
heartdata_treatment = heartdata.copy()
treatments_ar = heartdata_treatment['Treatment'].unique()

treatments = []
for treatment in treatments_ar: 
    treatment = treatment.split(',')
    treatments.append(treatment)

treatments_lst = []
for treatment in treatments:
    for element in treatment: 
        cleaned_element = element.lower().strip()
        if cleaned_element not in treatments_lst: 
            treatments_lst.append(cleaned_element)

treatments_lst.pop(39)
treatments_lst.pop(39)
treatments_lst.pop(32)
treatments_lst.pop(32)
treatments_lst.pop(32)
treatments_lst.append("Surgical options include valve replacement, valve repair, or removal of the infected valve tissue.")
treatments_lst.append("Pulmonary thromboendarterectomy (PTE), Balloon pulmonary angioplasty (BPA)")



treatments_cat = {}
i=0
for treatment in treatments_lst: 
    treatments_cat[treatment]=i
    i+=1


for treatment in treatments_cat: 
    heartdata_treatment[treatment]=0 



for treatment in treatments_cat: 
    i=0
    for treatments in heartdata_treatment["Treatment"]:
        treatments = treatments.split(",")
        for element in treatments: 
            cleaned_element = element.lower().strip()
            if cleaned_element in treatment:
                heartdata_treatment[treatment].iloc[i]=1
        i+=1
        

#assign unique labels 
treatments_ar = heartdata_treatment['Treatment'].unique()
i=0
treatments_label = {}
for treatment in treatments_ar: 
    treatments_label[treatment]=i
    i+=1

heartdata_treatment["Treatment Label"] = 0

for i in range(0, len(heartdata_treatment["Treatment"])):
  for treatment in treatments_label.keys(): 
      if treatment==heartdata_treatment["Treatment"].iloc[i]:
          heartdata_treatment["Treatment Label"].iloc[i]=treatments_label[treatment]
          break


#create vectors for treatment (hard labels)
heartdata_treatment = heartdata_treatment.astype('object')
#print(treatments_cat)

heartdata_treatment["Treatment Vector (Hard)"] = [[0]*38]*len(heartdata_treatment)

for i in range(len(heartdata_treatment)): 
  t = heartdata_treatment["Treatment"].iloc[i].lower()
  treatment_vector = []
  for treatment, val in treatments_cat.items():
    if treatment in t:  
      treatment_vector.append(1)
    else: 
      treatment_vector.append(0)
  heartdata_treatment["Treatment Vector (Hard)"][i]=treatment_vector


#convert hard labels to soft labels 
heartdata_treatment["Treatment Vector (Soft)"] = [[0]*38]*len(heartdata_treatment)
label_smoothing = 0.1 

one_to_smooth = (1-label_smoothing)+label_smoothing/38
zero_to_smooth = label_smoothing/38

for i in range(len(heartdata_treatment)): 
  t = heartdata_treatment["Treatment"].iloc[i].lower()
  treatment_vector = []
  for treatment, val in treatments_cat.items():
    if treatment in t:  
      treatment_vector.append(one_to_smooth)
    else: 
      treatment_vector.append(zero_to_smooth)
  heartdata_treatment["Treatment Vector (Soft)"][i]=treatment_vector


heartdata_treatment.to_excel("Treatment.xlsx")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
heartdata.describe()

#Remove Outliers for Age

def remove_outliers(df, df_col): 
    q1 = np.percentile(df_col, 25)
    q3 = np.percentile(df_col, 75)
    iqr = q3-q1
    low_bound = q1-(1.5*iqr)
    upp_bound = q3+(1.5*iqr)
    df=df[(df_col>=low_bound)&(df_col<=upp_bound)]
    return df

heartdata = remove_outliers(heartdata, heartdata["Age"])
heartdata.describe()

#Change "Heart Disease" column from string to numerical categorical data 
heartdata["Heart Disease"] = heartdata["Heart Disease"].map({"Absence":0, "Presence":1})
heartdata["Heart Disease"].value_counts()

heartdata["Gender"] = heartdata["Gender"].map({"Male":0, "Female":1})

heartdata["Blood culture"].value_counts()

"""
Sort strings into categorical data, categorize by type of bacteria: None(0), Staphylococcus(1), Streptococcus(2), 
Candida(3), Other(4)
"""
def blood_culture(data): 
    if data=='None': 
        return 0
    elif 'Staphylococcus' in data: 
        return 1
    elif 'Streptococcus' in data: 
        return 2 
    elif 'Candida' in data: 
        return 3
    else: 
        return 4 
    
heartdata["Blood culture"]=heartdata["Blood culture"].map(blood_culture)
heartdata["Blood culture"].value_counts()

heartdata["Echocardiogram"].value_counts() #this data is partially represented in other columns - drop?

heartdata["EKG"].value_counts() #too many variations - drop col?

heartdata["Cardiac CT"].value_counts()

heartdata["Chest x-ray"].value_counts() #too many empty, drop col or assume normal lung structure?

heartdata["Previous illnesses"].value_counts() #change to 1s and 0s

def previous_illnesses(data): 
    if data=='None': 
        return 0
    else:
        return 1
    
heartdata["Previous illnesses"]=heartdata["Previous illnesses"].map(previous_illnesses)
heartdata["Previous illnesses"].value_counts()

heartdata["Pulmonary function tests"].value_counts() #too many empty, drop col

heartdata["Spirometry"].value_counts() #too many empty, drop col

def stenosis(data): 
    if data["Mitral stenosis"]==1: 
        return 1
    elif data["Aortic stenosis"]==1: 
        return 1 
    elif data["Tricuspid stenosis"]==1:
        return 1 
    elif data["Pulmonary stenosis"]==1: 
        return 1
    else: 
        return 0
    
heartdata["Stenosis"]=heartdata.apply(lambda data:stenosis(data), axis=1)

def cardiomyopathy(data): 
    if data["Dilated cardiomyopathy"]==1: 
        return 1
    elif data["Hypertrophic cardiomyopathy"]==1: 
        return 1 
    elif data["Restrictive cardiomyopathy"]==1:
        return 1 
    elif data["Arrhythmogenic right ventricular cardiomyopathy"]==1: 
        return 1
    elif data["Takotsubo cardiomyopathy"]==1: 
        return 1
    else: 
        return 0
heartdata["Cardiomyopathy"]=heartdata.apply(lambda data:cardiomyopathy(data), axis=1)

heartdata.to_excel("CleanHeartData.xlsx")
