# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle 

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Richa Pawar/Documents/ML project/Diabetes Prediction/diabetes_trained_model .sav', 'rb'))


input_data = (1,89,76,34,37,31.2,0.192,23)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
  print("The Person is not Diabetic")
else:
  print("The Person is Diabetic")