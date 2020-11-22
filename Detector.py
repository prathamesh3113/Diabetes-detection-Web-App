#Description: This programs detects if someone has diabetes using machine learning and python

#Impot Libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")
# adding color as green
st.markdown(
    """                                                                     
    <style>                                                                 
    .sidebar .sidebar-content {                                             
        background-image: linear-gradient(#053552,#053552);                 
        color: white;                                                       
    }                                                                       
    </style>                                                                
    """, unsafe_allow_html=True,
)


#Create a title and Sub-title
st.write(" # Diabetes Detector System ")
st.subheader("if someone has diabetes using machine learning and python !")
#Open and Display an Image
image = Image.open("C:/Users/Vikas/PycharmProject/CreditCardDetection/Cancer.JPG")
st.image(image , caption="ML",use_column_width=True)

#Get the Data
df = pd.read_csv('C:/Users/Vikas/PycharmProject/CreditCardDetection/diabetes.csv')
#Set a subheader
st.markdown("<h2 style='text-align: left; color: white ;'> Data Information Of Pima Indians Diabetes Database </h2>",unsafe_allow_html=True)
#Show the data as a Table
st.dataframe(df)
st.markdown("<h2 style='text-align: left; color: white ;'>Summary Of Dataset </h2>",unsafe_allow_html=True)
#show the statistics on the data
st.write(df.describe().T)
st.markdown("<h3 style='text-align: left; color: white ;'>Columns Of Dataset </h3>",unsafe_allow_html=True)
st.write(df.columns)
st.markdown("<h3 style='text-align: left; color: white ;'>Shape Of Dataset : </h3>",unsafe_allow_html=True)
st.write(df.shape)
# No of valid cases in dataset
st.markdown("<h3 style='text-align:left ; color: white;'>Valid Diabetes Person :</h3>", unsafe_allow_html=True)
st.write(' {}'.format(len(df[df['Outcome'] == 0])))
# No of fraud cases in dataset
st.markdown("<h3 style='text-align:left ; color: white;'>Non-Diabetes Person: </h3>",unsafe_allow_html=True)
st.write('{}'.format(len(df[df['Outcome'] == 1])))

st.markdown("<h2 style='text-align: left; color: white ;'>Visulization Given Dataset </h2>",unsafe_allow_html=True)
#Show the  data as a chart
chart = st.bar_chart(df)

#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:,-1].values

#Split the data set into 75% Traning and 25% Testing Set
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.25 , random_state=0)

#Get the feature input from the user
def get_user_input():
    st.sidebar.markdown("<h3 style='text-align: left; color: white ;'>Select following Parameteres According to your condition </h3>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='text-align: left; color: white ;'>Pregnancies</h3>", unsafe_allow_html=True)
    pregnancies = st.sidebar.slider('pregnancies', 0 , 17 , 3)
    st.sidebar.markdown("<h3 style='text-align: left; color: white ;'>Glucose</h3>", unsafe_allow_html=True)
    glucose = st.sidebar.slider('glucose', 0 , 199 ,117)
    st.sidebar.markdown("<h3 style='text-align: left; color: white ;'>Blood Pressure</h3>", unsafe_allow_html=True)
    blood_pressure = st.sidebar.slider('blood_pressure', 0 , 122 ,72)
    st.sidebar.markdown("<h3 style='text-align: left; color: white ;'>Skin Thickness</h3>", unsafe_allow_html=True)
    skin_thickness = st.sidebar.slider('skin_thickneess', 0 , 99 ,23)
    st.sidebar.markdown("<h3 style='text-align: left; color: white ;'>Insulin</h3>", unsafe_allow_html=True)
    insulin  =st.sidebar.slider('insulin', 0.0 , 846.0 ,30.0)
    st.sidebar.markdown("<h3 style='text-align: left; color: white ;'>BMI</h3>", unsafe_allow_html=True)
    bmi = st.sidebar.slider('bmi', 0.0 , 67.1 ,32.0)
    st.sidebar.markdown("<h3 style='text-align: left; color: white ;'>Diabetes Pedigree Function</h3>", unsafe_allow_html=True)
    diabetes_pedigree_function =st.sidebar.slider('diabetes_pedigree_funcation', 0.078 , 2.42 ,0.3725)
    st.sidebar.markdown("<h3 style='text-align: left; color: white ;'>Age</h3>", unsafe_allow_html=True)
    age = st.sidebar.slider('age', 21 , 81 ,29)

    #Store a dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness ': skin_thickness,
                 'insulin':insulin ,
                 'bmi': bmi ,
                 'diabetes_pedigree_function':diabetes_pedigree_function,
                 'age': age
                 }
    #Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features
#Store the user input a variable
user_input = get_user_input()

#Set a subheader and display the user input
st.subheader("User Input:")
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train , Y_train)

#Show the models metrics
st.subheader("Model Test Accuracy Score :")
st.write( str (accuracy_score(Y_test ,RandomForestClassifier.predict(X_test))*100 )+ '%')

#Store the models predication in a variable
prediction = RandomForestClassifier.predict(user_input)

#Set the subheader the Classification
st.subheader("Classification of given user input :")
st.write(prediction)

st.markdown("<h2 style='text-align: left; color: white ;'>Conclusion :</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left; color: white ;'>Glucose’ and ‘BMI’ are the"
                    " most important medical predictor features.For a healthy living, look after your sugar intake and your weight.""</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left; color: white ;'>I wish all a healthy life!</h2>", unsafe_allow_html=True)
