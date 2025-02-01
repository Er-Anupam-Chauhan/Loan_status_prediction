import numpy as np
import pickle
import streamlit as st


#Loading the saved model 
loded_model = pickle.load(open('trained_model.sav','rb'))

def prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loded_model.predict(input_data_reshaped) # using loaded model 
    print(prediction)

    if (prediction[0] == 1):
        return 'The Loan can be provided'
    else:
        return 'The loan can not be provided'
    
def main():

    #giving a title
    st.title("Loan Prediction Web App")

    #getting input data from user
    Gender = st.text_input("Enter Gender : ")
    Married = st.text_input("Enter Marratial status : ")
    Dependents = st.text_input("Enter number of Dependents : ")
    Education = st.text_input("Enter Education : ")
    Self_Employed = st.text_input("Enter Self Employed or not: ")
    ApplicantIncome = st.text_input("Enter Applicants Income : ")
    CoapplicantIncome = st.text_input("Enter Co-Applicants Income : ")
    LoanAmount = st.text_input("Enter Loan Amount : ")
    Loan_Amount_Term = st.text_input("Enter Loan Amount Term : ")
    Credit_History = st.text_input("Enter Credit History : ")
    Property_Area = st.text_input("Enter Property Area : ")

    #code for prediction
    Approval_status = '' #for function output to store

    #Creating a button for prediction
    if st.button('Check Approval status'):
        Approval_status = prediction([Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area])
    st.success(Approval_status)


if __name__ == '__main__':
    main()

