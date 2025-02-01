import numpy as np
import pickle

#Loading the saved model 
loded_model = pickle.load(open('trained_model.sav','rb'))

input_data = (1, 1, 0, 1, 0, 2400, 2167.0, 115.0, 360.0, 1.0, 1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loded_model.predict(input_data_reshaped) # using loaded model 
print(prediction)

if (prediction[0] == 1):
  print('The Loan can be provided')
else:
  print('The loan can not be provided')