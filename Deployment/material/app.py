from flask import Flask,render_template,request, flash
import joblib
from helpers.dummies1_py import *

app=Flask(__name__)

model=joblib.load('models/model.h5')
scaler=joblib.load('models/scaler.h5')

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    all_data=request.args
    Mileage=float(all_data['Mileage'])
    Engine=float(all_data['Engine'])
    Power=float(all_data['Power'])
    Tax=float(all_data['Tax'])
    cars_old=float(all_data['carsold'])  
    miles_driven=float(all_data['miles_driven'])       
    Owner_Type=all_data['Owner_Type']
    Location=all_data['location']
    Fuel_Type=all_data['Fuel_Type']
    Transmission=all_data['transmission']
    Seats=all_data['seats']


    data=[Mileage,Engine,Power,Tax,cars_old,miles_driven]+owner_type_dummies[Owner_Type]+Location_dummies[Location]+Fuel_Type_dummies[Fuel_Type]+Transmission_dummies[Transmission]+seats_dummies[Seats]
    final_data=scaler.transform([data])
    pred=model.predict(final_data)[0]

    return render_template('prediction.html',price=pred)

if __name__=='__main__':
    app.run()