# -*- coding: utf-8 -*-
"""dummies1.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LjhFB7C-AzqRHaGGUiPyW86isSy_Z7Hc
"""

owner_type_dummies = {
    'first':[0,0,0,0],
    'second':[0,1,0,0],
    'third':[0,0,1,0],
    'fourth':[1,0,0,0]
}

Location_dummies = {
    'Bangalore': [1,0,0,0,0,0,0,0,0,0,0],
    'Chennai': [0,1,0,0,0,0,0,0,0,0,0],
    'Coimbatore': [0,0,1,0,0,0,0,0,0,0,0],
    'Delhi': [0,0,0,1,0,0,0,0,0,0,0],
    'Hyderabad':[0,0,0,0,1,0,0,0,0,0,0],
    'Jaipur':[0,0,0,0,0,1,0,0,0,0,0],
    'Kochi':[0,0,0,0,0,0,1,0,0,0,0],
    'Kolkata':[0,0,0,0,0,0,0,1,0,0,0],
    'Mumbai':[0,0,0,0,0,0,0,0,1,0,0],
    'Pune':[0,0,0,0,0,0,0,0,0,1,0],
    'Ahmedabad':[0,0,0,0,0,0,0,0,0,0,0]
}

Fuel_Type_dummies = {
    'Diesel': [1, 0, 0,0],
    'LPG': [0, 1, 0,0],
    'Petrol': [0, 0,1,0],
    'CNG':[0,0,0,0]
}

seats_dummies = {
    '4': [0, 0, 0,0,0,0,0],
    '5': [1, 0, 0,0,0,0,0],
    '6': [0, 1, 0,0,0,0,0],
    '7': [0, 0, 1,0,0,0,0],
    '8': [0, 0, 0,1,0,0,0],
    '9': [0, 0, 0,0,1,0,0],
    '10': [0, 0, 0,0,0,1,0]
}
Transmission_dummies= {
    'manual':[1,0],
    'automatic':[0,0]
}