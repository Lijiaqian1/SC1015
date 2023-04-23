from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, Form, SelectField, IntegerField
from wtforms.validators import InputRequired, Length, NumberRange, DataRequired
from flask_bootstrap import Bootstrap5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
bootstrap = Bootstrap5(app)

app.config['SECRET_KEY'] = 'paRxT4w@J+c*5mS&SjTZ-58?=ugB4=@T'

#Form class
class PatientDataEntry(FlaskForm):
    sex = SelectField("Sex:", choices=[("0","Male"),("1","Female")])
    age = IntegerField("Age: ")
    chest_pain = BooleanField("Chest Pain: ")
    shortness_breath = BooleanField("Shortness of breath: ")
    fatigue = BooleanField("Fatigue: ")
    systolic = IntegerField("Systolic (mmHg): ")
    diastolic = IntegerField("Diastolic (mmHg): ")
    heart_rate = IntegerField("Heart rate (bpm): ")
    lung_sound = BooleanField("Lung sounds: ")
    cholesterol_level = IntegerField("Cholesterol level (mg/dL): ")
    ldl = IntegerField("LDL level (mg/dL): ")
    hdl = IntegerField("HDL level (mg/dL): ")
    diabetes = BooleanField("Diabetes:")
    atrial_fibrillation = BooleanField("Atrial fibrillation:")
    mitral_valve_prolapse = BooleanField("Mitral valve prolapse:")
    rheumatic_fever = BooleanField("Rheumatic fever:")
    mitral_stenosis = BooleanField("Mitral stenosis:")
    aortic_stenosis = BooleanField("Aortic stenosis:")
    tricuspid_stenosis = BooleanField("Tricuspid stenosis:")
    pulmonary_stenosis = BooleanField("Pulmonary stenosis:")
    dc = BooleanField("Dilated cardiomyopathy:")
    hc = BooleanField("Hypertrophic cardiomyopathy:")
    rc = BooleanField("Restrictive cardiomyopathy:")
    arvc = BooleanField("Arrhythmogenic right ventricular cardiomyopathy:")
    tc = BooleanField("Takotsubo cardiomyopathy:")
    du = BooleanField("Drug use:")
    fever = BooleanField("Fever:")
    chills = BooleanField("Chills:")
    jp = BooleanField("Joint pain:")
    alcoholism = BooleanField("Alcoholism:")
    hypertension = BooleanField("Hypertension:")
    fainting = BooleanField("Fainting:")
    dizziness = BooleanField("Dizziness:")
    smoking = BooleanField("Smoking:")
    hc = BooleanField("High cholesterol:")
    echocardiogram = StringField("Echocardiogram comments:")
    bc = StringField("Blood culture comments:")
    ekg = StringField("Electrocardiogram comments:")
    cct = StringField("Cardiac CT comments:")
    obesity = BooleanField("Obesity:")
    murmur = BooleanField("Murmur:")
    cx = StringField("Chest x-ray comments:")
    pi = BooleanField("Previous illnesses:")
    pft = StringField("Pulmonary function tests comments:")
    spirometry = StringField("Spirometry comments:")

class HeartDiseaseData():
    def __init__(self) -> None:
        self.heartdata = pd.read_excel("CleanHeartData.xlsx")
        self.target = self.heartdata['Heart Disease']
        self.data = self.heartdata.drop(['Echocardiogram','EKG','Cardiac CT','Chest x-ray','Pulmonary function tests','Spirometry','Medications','Treatment','Heart Disease'],axis=1)
        self.heartdata_treatment, self.treatments_lst = self.process_treatment()
         

    def xgb_model(self, tobepredicted):
        tobepredicted = np.array(tobepredicted).reshape(1, -1)
        #app.logger.info("data.head: {}".format(data.head()))
        standardScaler = StandardScaler()
        standardScaler.fit(self.data)
        self.data =  standardScaler.transform(self.data)
        tobepredicted = standardScaler.transform(tobepredicted)
        train_X,test_X,train_y,test_y = train_test_split(self.data,self.target,random_state=3)

        """### XGBoost with PCA"""

        # define the pipeline with PCA and XGBoost
        pipeline = Pipeline([
            ('pca', PCA(n_components=21)),
            ('xgb', XGBClassifier())
        ])

        # define the hyperparameters for tuning
        parameters = {
            'xgb__n_estimators': [50, 100, 200],
            'xgb__max_depth': [3, 5, 7]
        }

        # perform grid search with cross-validation
        grid_search = GridSearchCV(pipeline, parameters, cv=5)
        grid_search.fit(train_X, train_y)

        # print the best parameters and score
        print("Best parameters: ", grid_search.best_params_)
        print("XGB Best score: ", grid_search.best_score_)

        y_pred = grid_search.best_estimator_.predict(test_X)

        accuracy = accuracy_score(test_y, y_pred)

        print("XGB KNN Accuracy:", accuracy)

        prediction = grid_search.best_estimator_.predict(tobepredicted)
        
        return prediction, accuracy
    
    def KNN_model(self, tobepredicted):
        tobepredicted = np.array(tobepredicted).reshape(1, -1)
        #app.logger.info("data.head: {}".format(data.head()))
        standardScaler = StandardScaler()
        standardScaler.fit(self.data)
        self.data =  standardScaler.transform(self.data)
        tobepredicted = standardScaler.transform(tobepredicted)
        train_X,test_X,train_y,test_y = train_test_split(self.data,self.target,random_state=3)


        knn =  KNeighborsClassifier()
        knn.fit(train_X,train_y)

        knn_pred_y = knn.predict(test_X)

        prediction = knn.predict(tobepredicted)

        accuracy = accuracy_score(test_y,knn_pred_y)

        return prediction, accuracy
    
    def process_treatment(self):
        #create columns for each treatment 
        heartdata_treatment = self.heartdata.copy()

        treatments_ar = heartdata_treatment['Treatment'].unique()

        #identify individual treatments in each row
        treatments = []  
        for treatment in treatments_ar: 
            treatment = treatment.split(',')
            treatments.append(treatment)

        #create list of unique treatments
        treatments_lst = [] 
        for treatment in treatments:
            for element in treatment: 
                cleaned_element = element.lower().strip()
                if cleaned_element not in treatments_lst: 
                    treatments_lst.append(cleaned_element)

        #account for exceptions that wrongly separates treatments by comma 
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

        #create individual columns using cleaned data 
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



        heartdata_treatment["Treatment Label"]=0

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

        print(heartdata_treatment[["Treatment", "Treatment Vector (Hard)"]]) 

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

        print(heartdata_treatment[["Treatment", "Treatment Vector (Soft)"]]) 
        return heartdata_treatment, treatments_lst

    def treatment_model(self, tobepredicted):
        treatment_vectors = self.heartdata_treatment['Treatment Vector (Soft)']
        tobepredicted = np.array(tobepredicted).reshape(1, -1)

        X_train, X_test, y_train, y_test = train_test_split(self.data, treatment_vectors, test_size=0.2, random_state=42)


        regressor = MultiOutputRegressor(LinearRegression())

        # convert y_train to a 2D array
        y_train = np.vstack(y_train)
        y_test = np.vstack(y_test)
        # fit the model to the training data
        regressor.fit(X_train, y_train)


        y_pred = regressor.predict(X_test)

        result = regressor.predict(tobepredicted)
        
        # evaluate the model's performance
        mse = mean_squared_error(y_test, y_pred)  # mean squared error

        print("Mean Squared Error: ", mse)
        for i in range(len(y_test)):
            
            print("Actual target column: ", y_test[i])
            print("Predicted target column: ", y_pred[i])
            print("---------------")

        #Convert vector columns back to strings for multioutput regression results  
        actual_list = []
        predicted_list = []
        for i in range(len(y_test)):
            print("Actual treatment:", i)
            actual_list.append([])
            predicted_list.append([])
            for j in range(38):
                if y_test[i][j]>=0.2: 
                    print(self.treatments_lst[j])
                    actual_list[i].append(self.treatments_lst[j])

            print("\nPredicted treatment:")
            for j in range(38):
                if y_pred[i][j]>=0.2: 
                    print(self.treatments_lst[j])
                    predicted_list[i].append(self.treatments_lst[j])

            print("---------------")

        correct_num = []
        wrong_num  = []
        multiregAcc = 0
        multiregwrong = 0
        for i in range(len(actual_list)):
            correct_num.append(0)
            wrong_num.append(0)
            temp = set(actual_list[i]) & set(predicted_list[i])
            for j in temp:
                correct_num[i] += 1
            temp = set(predicted_list[i]) - (set(actual_list[i]) & set(predicted_list[i]))
            for j in temp:
                wrong_num[i] += 1
        for i in correct_num:
            multiregAcc += i
        multiregAcc /= sum(len(l) for l in actual_list)
        for i in wrong_num:
            multiregwrong += i
        multiregwrong /= sum(len(l) for l in predicted_list)
        print(f"Multi-output regression hit rate: {multiregAcc}")
        print(f"Multi-output regression miss rate: {multiregwrong}")
    
hdata = HeartDiseaseData() #init data

def map_blood_culture(data):
    if data=="": 
        return 0
    data = str(data)
    data = data.strip()
    
    if 'Staphylococcus' in data: 
        return 1
    elif 'Streptococcus' in data: 
        return 2 
    elif 'Candida' in data: 
        return 3
    else: 
        return 4 
    
def map_true_false(data):
    if isinstance(data,bool):
        if data == False:
            return 0
        elif data == True:
            return 1
        else:
            return data
    else:
        return data
    
    

@app.route('/', methods=['GET', 'POST'])
def index():
    form = PatientDataEntry()
    
    if form.validate_on_submit():
        app.logger.info("Form recieved successfully")
            
        # process data
        app.logger.info(form.bc.data=="")
        form.bc.data = map_blood_culture(form.bc.data) 
        stenosis = form.mitral_stenosis.data or form.aortic_stenosis.data or form.tricuspid_stenosis.data or form.pulmonary_stenosis.data
        cardiomyopathy = form.dc.data or form.rc.data or form.arvc.data or form.tc.data or form.hc.data

        # # Add data to Excel sheet
        
        row_value = [form.sex.data, form.age.data, form.chest_pain.data,form.shortness_breath.data,
               form.fatigue.data,form.systolic.data,form.diastolic.data,form.heart_rate.data,
               form.lung_sound.data,form.cholesterol_level.data,form.ldl.data,form.hdl.data,
               form.diabetes.data,form.atrial_fibrillation.data,form.mitral_valve_prolapse.data,
               form.rheumatic_fever.data,form.mitral_stenosis.data,form.aortic_stenosis.data,
               form.tricuspid_stenosis.data,form.pulmonary_stenosis.data,form.dc.data,form.hc.data,
               form.rc.data,form.arvc.data,form.tc.data,form.du.data,form.fever.data,form.chills.data,
               form.jp.data,form.alcoholism.data,form.hypertension.data,form.fainting.data,
               form.dizziness.data,form.smoking.data,form.hc.data,form.bc.data,
               form.obesity.data,form.murmur.data,form.pi.data,stenosis,cardiomyopathy]
        row_value = map(map_true_false,row_value)
        row_value = list(row_value)
        app.logger.info(row_value)
        predicted_value, confidence = hdata.KNN_model(row_value)
        app.logger.info(f"predicted_value: {predicted_value}, confidence: {confidence}")

        return render_template('result.html',predicted_value=predicted_value[0],confidence=confidence)
    elif request.method == 'POST':
        app.logger.warn("Validation error! : {}".format(form.errors))
        return redirect("/")
    else:
        return render_template('index.html',form=form)

if __name__ == '__main__':
    app.run(debug=True)