from flask import Flask,render_template,request
import joblib
from helpers.dummies1_py import *

app=Flask(__name__)

model=joblib.load('material/models/model.h5')
scaler=joblib.load('material/models/scaler.h5')

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    Mileage=float(request.form['Mileage'])
    Engine=float(request.form['Engine'])
    Power=float(request.form['Power'])
    Tax=float(request.form['Tax'])
    cars_old=float(request.form['carsold'])  
    miles_driven=float(request.form['miles_driven'])       
    Owner_Type=request.form['Owner_Type']
    Location=request.form['location']
    Fuel_Type=request.form['Fuel_Type']
    Transmission=request.form['transmission']
    Seats=request.form['seats']

    
    data=[Mileage,Engine,Power,Tax,cars_old,miles_driven]+owner_type_dummies[Owner_Type]+Location_dummies[Location]+Fuel_Type_dummies[Fuel_Type]+Transmission_dummies[Transmission]+seats_dummies[Seats]
    final_data=scaler.transform([data])
    pred=model.predict(final_data)[0]

    return render_template('prediction.html',price=pred)

if __name__=='__main__':
    app.run()