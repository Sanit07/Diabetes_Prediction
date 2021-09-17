from flask import Flask,request,render_template
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if  request.method == "POST":
        Pregnancies=request.form['Pregnancies']
        Glucose=request.form['Glucose']
        BloodPressure=request.form['BloodPressure']
        SkinThickness=request.form['SkinThickness']
        Insulin=request.form['Insulin']
        BMI=float(request.form['BMI'])
        DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
        Age=request.form['Age']
        output=model.predict(np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
        BMI, DiabetesPedigreeFunction, Age]]))
        if output[0] > 0:
            return render_template('result.html',Prediction=f'You are diabetic :(')
        else:
            return render_template('result.html',Prediction=f'You are not diabetic :)')

    else:
        return render_template('index.html')




if __name__=="__main__":
    app.run(debug=True)