import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import pickle
import shutil
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots
from PIL import Image, ImageOps
import numpy as np
import base64
import path
import json
import streamlit.components.v1 as components
from state_district_data import states, districts, seasons, crops

import tensorflow as tf
from tensorflow.keras.models import load_model

# pickle_in1 = open("Agri_pesticide.pkl","rb")
# model1=pickle.load(pickle_in1)

# pickle_in2 = open("Agri_pesticide.pkl","rb")
# model2=pickle.load(pickle_in2)

# pickle_in3 = open("Agri_pesticide.pkl","rb")
# model3=pickle.load(pickle_in3)



pickle_in4 = open("cropRecommender.pkl","rb")
model4=pickle.load(pickle_in4)

model5 = load_model('plant_disease.hdf5')



pickle_in6 = open("Agri_pesticide.pkl","rb")
model6=pickle.load(pickle_in6)
def predict_crop_damage(Estimated_Insects_Count,Crop_Type,Soil_Type,Pesticide_Use_Category,Number_Doses_Week,Number_Weeks_Used,Number_Weeks_Quit,Season):

    prediction=model6.predict([[Estimated_Insects_Count,Crop_Type,Soil_Type,Pesticide_Use_Category,Number_Doses_Week,Number_Weeks_Used,Number_Weeks_Quit,Season]])
    print(prediction)
    return prediction

def import_and_predict(image_data,model):
    img=ImageOps.fit(image_data, size=(220,220))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x/255
    result = model.predict([np.expand_dims(x, axis=0)])
    
    return result

def crop_price_prediction(input_list):
    ohe = {}
    pickle_file = open("ohe.pkl","rb")
    ohe = pickle.load(pickle_file)
    model = load_model('crop_price_prediction.h5')
    input_array =  np.array(input_list).reshape(-1,5)
    encoded_input = ohe.transform(input_array)
    prediction = model.predict(encoded_input)
    prediction = float(prediction)

    return prediction

def yield_prediction(input_list_yield):
    try:
        # Create DataFrame with correct column names
        input_df = pd.DataFrame([input_list_yield], 
                            columns=['State_Name', 'District_Name', 'Crop_Year', 
                                    'Season', 'Crop', 'Area'])
        
        # Load the encoder and model
        with open("oneHotEncoder.pkl", "rb") as f:
            ohe = pickle.load(f)
            
        with open("crop_yield_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Extract features for encoding
        categorical_features = ['State_Name', 'District_Name', 'Season', 'Crop']
        features_to_encode = input_df[categorical_features]
        
        # One-hot encode the categorical features
        encoded_features = ohe.transform(features_to_encode)
        
        # Create the final feature matrix with encoded features and Area
        input_df['Area'] = pd.to_numeric(input_df['Area'], errors='coerce')
        X = np.hstack([encoded_features, input_df[['Area']].values])
        
        # Make prediction (output will be in kg/hectare)
        prediction = model.predict(X)[0]
        
        # Ensure prediction is non-negative and realistic
        prediction = max(0, min(prediction, 100000))  # Cap at 100 tonnes/ha
        
        return float(prediction)
        
    except Exception as e:
        print(f"Error in yield_prediction: {str(e)}")
        return None

def img_to_bytes(img_path):
    img_bytes = path.Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

#@st.cache(allow_output_mutation=True)

def main():

    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"><font size="10"> AGRI AI 🌱 </font></h1>
    </div>
    <style>
    .reportview-container .main{
        background: url("https://i.ibb.co/Sny7Wzm/Mid-shade.png");
        background-size: cover;
    }
   .sidebar .sidebar-content {
        background: url("https://i.ibb.co/Sny7Wzm/Mid-shade.png")
    }
    </style>
    <style>
    body {
    background-image: url("https://i.ibb.co/Sny7Wzm/Mid-shade.png");
    background-size: cover;
    }
    </style>
    """
    
    
    st.markdown(html_temp, unsafe_allow_html=True)


    # image = Image.open('logo1.jpg')
    # st.image(image, caption=None, width=200, use_column_width=None, clamp=True, channels='RGB', output_format='auto')

    selection = st.radio("", ['Crop Disease Detection', 'Crop Recommendation','Yield Prediction', 'Crop Price Recommendation', 'Crop Health'])
    st.write("""<style>
            .reportview-container .markdown-text-container {
                font-family: monospace;
            }
            .sidebar .sidebar-content {
                background-image: linear-gradient(#FFFFFF,#FFFFFF);
                color: white;
            }
            .Widget>label {
                color: white;
                font-family: monospace;
            }
            [class^="st-b"]  {
                color: white;
                font-family: monospace;
            }
            .st-bb {
                background-color: transparent;
            }
            .st-at {
                
            }
            footer {
                font-family: monospace;
            }
            .reportview-container .main footer, .reportview-container .main footer a {
                color: #FFFFFF;
            }
            header .decoration {
                background-image: none;
            }

            </style>""", unsafe_allow_html=True)

    if selection == 'Crop Health':
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Health</b></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(textbg.format(bgcolor,fontcolor),unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b> A good harvest is ensured by several factors such as availability of water, soil fertility, protecting crops from rodents, timely use of pesticides & other useful chemicals and nature. While a lot of these factors are difficult to control for, the amount and frequency of pesticides is something the farmer can control.</b></font></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(text.format(bgcolor,fontcolor),unsafe_allow_html=True)

        st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
         unsafe_allow_html=True,)
        Estimated_Insects_Count = st.sidebar.text_input("Estimated Insects Count per square meter","Type Here")
        Crop_Type = st.sidebar.selectbox("Select your crop type",("Food Crop", "Cash Crop"))
        Soil_Type = st.sidebar.selectbox("Select your soil type",("Dry", "Wet"))
        Pesticide_Use_Category = st.sidebar.selectbox("Pesticide Use Category",("Never", "Previously Used", "Currently Using"))
        Number_Doses_Week = st.sidebar.text_input("Number Doses per Week","Type Here")
        Number_Weeks_Used = st.sidebar.text_input("Number of Weeks Used","Type Here")
        Number_Weeks_Quit = st.sidebar.text_input("Number of Weeks Quit","Type Here")
        Season = st.sidebar.selectbox("Season",("Kharif", "Rabi", "Zaid"))

        if Crop_Type == 'Food Crop':
            Crop_Type = 0
        else :
            Crop_Type = 1

        if Soil_Type == 'Dry':
            Soil_Type = 0
        else :
            Soil_Type = 1

        if Pesticide_Use_Category == "Never":
            Pesticide_Use_Category = 1 
        elif Pesticide_Use_Category == "Previously Used":
            Pesticide_Use_Category = 2
        else :
            Pesticide_Use_Category = 3

        if Season == "Kharif":
            Season = 1 
        elif Season == "Rabi":
            Season = 2
        else :
            Season = 3

        

        result=""
        if st.button("Predict"):
            result=predict_crop_damage(Estimated_Insects_Count,Crop_Type,Soil_Type,Pesticide_Use_Category,Number_Doses_Week,Number_Weeks_Used,Number_Weeks_Quit,Season)
        if result==0:
            result = 'Alive'
        elif result==1:
            result = 'Possible Damage due to other causes'
        elif result==2:
            result = 'Damage due to Pesticides'

        st.success('The crop is {}'.format(result))

    elif selection =='Crop Disease Detection':
   
        #st.set_option('depreciation.showfileUploaderEncoding',False)
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Diseases Detection</b></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(textbg.format(bgcolor,fontcolor),unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b> In recent times, drastic climate changes and lack of immunity in crops has caused substantial increase in growth of crop diseases. This causes large scale demolition of crops, decreases cultivation and eventually leads to financial loss of farmers. Due to rapid growth in variety of diseases , identification and treatment of the disease is a major importance.</b></font></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(text.format(bgcolor,fontcolor),unsafe_allow_html=True)

        st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
         unsafe_allow_html=True,)
        
        file=st.sidebar.file_uploader("Please upload a crop image")
        
        #text_io = io.TextIOWrapper(file)
        if st.button("Detect"):
            if file is None:
                st.sidebar.text("please upload an image file")
                
            else:
                image=Image.open(file)
                st.image(image,use_column_width=True)
                predictions=import_and_predict(image,model5)
                file_json = open("class_indices.json", "r")
                class_indices = json.load(file_json)
                classes=list(class_indices.keys())
                classresult=np.argmax(predictions,axis=1)
                word=classes[classresult[0]].split("__")
                word[0] = word[0].replace("_"," ")
                word[1] = word[1].replace("_"," ")
                st.success("This crop is {} and it has {} ".format(word[0], word[1]))

    elif selection == 'Crop Recommendation':
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Recommendation</b></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(textbg.format(bgcolor,fontcolor),unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b> Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.</b></font></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(text.format(bgcolor,fontcolor),unsafe_allow_html=True)

        st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
         unsafe_allow_html=True,)
        
        st.sidebar.markdown(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.sidebar.number_input("Nitrogen", 1,10000)
        P = st.sidebar.number_input("Phosporus", 1,10000)
        K = st.sidebar.number_input("Potassium", 1,10000)
        temp = st.sidebar.number_input("Temperature",0.0,100000.0)
        humidity = st.sidebar.number_input("Humidity in %", 0.0,100000.0)
        ph = st.sidebar.number_input("Ph", 0.0,100000.0)
        rainfall = st.sidebar.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            prediction = model4.predict(single_pred)
            
            st.success(f"{prediction.item().title()} are recommended by the A.I for your farm.")


    elif selection == 'Crop Price Recommendation':
        # st.sidebar.markdown('<p>Graphs are auto-updated on each change.</p>', unsafe_allow_html=True)
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Price Recommendation</b></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(textbg.format(bgcolor,fontcolor),unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b>Crop price forecasting techniques are essential because they enable supply chain planners and government bodies to take appropriate actions by estimating market factors such as demand and supply. In emerging economies such as India, the crop prices at marketplaces are manually entered every day, prone to human-induced errors like incorrect data or access to no data for many days. In addition to such human errors, the fluctuations in the prices themselves make creating a stable and robust forecasting solution a necessity.</b></font></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(text.format(bgcolor,fontcolor),unsafe_allow_html=True)

        st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
         unsafe_allow_html=True,)
        list1 = []
        state_selection = st.sidebar.selectbox("Select Your State",('Andaman and Nicobar', 'Andhra Pradesh', 'Assam', 'Chattisgarh',
       'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
       'Jammu and Kashmir', 'Karnataka', 'Kerala', 'Madhya Pradesh',
       'Maharashtra', 'Manipur', 'Meghalaya', 'Nagaland', 'Odisha',
       'Pondicherry', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Telangana',
       'Tripura', 'Uttar Pradesh', 'Uttrakhand', 'West Bengal'))

        district_selection = st.sidebar.selectbox("Select Your State", ('South Andaman', 'Chittor', 'Kurnool', 'West Godavari', 'Cachar',
       'Darrang', 'Dhubri', 'Jorhat', 'Kamrup', 'Sonitpur', 'Bastar',
       'Kanker', 'Surajpur', 'North Goa', 'Amreli', 'Anand', 'Bharuch',
       'Kachchh', 'Kheda', 'Panchmahals', 'Surat', 'Vadodara(Baroda)',
       'Valsad', 'Ambala', 'Faridabad', 'Gurgaon', 'Kurukshetra', 'Mewat',
       'Panipat', 'Kangra', 'Kullu', 'Badgam', 'Bangalore', 'Kolar',
       'Tumkur', 'Alappuzha', 'Ernakulam', 'Kannur', 'Kasargod', 'Kollam',
       'Kottayam', 'Malappuram', 'Thirssur', 'Thiruvananthapuram',
       'Anupur', 'Badwani', 'Dhar', 'Dindori', 'Jhabua', 'Khandwa',
       'Narsinghpur', 'Sheopur', 'Ahmednagar', 'Buldhana', 'Jalgaon',
       'Kolhapur', 'Nagpur', 'Nanded', 'Nashik', 'Pune', 'Satara',
       'Sholapur', 'Bishnupur', 'Chandel', 'Imphal East', 'Imphal West',
       'Thoubal', 'East Khasi Hills', 'Mokokchung', 'Angul', 'Balasore',
       'Bargarh', 'Dhenkanal', 'Gajapati', 'Ganjam', 'Jharsuguda',
       'Mayurbhanja', 'Nowarangpur', 'Sundergarh', 'Karaikal', 'Amritsar',
       'Barnala', 'Bhatinda', 'Gurdaspur', 'Hoshiarpur', 'Jalandhar',
       'Ludhiana', 'Mansa', 'Muktsar', 'Patiala', 'Ropar (Rupnagar)',
       'Tarntaran', 'Baran', 'Barmer', 'Bikaner', 'Chittorgarh', 'Jalore',
       'Jhalawar', 'Kota', 'Sikar', 'Tonk', 'Coimbatore', 'Cuddalore',
       'Dharmapuri', 'Dindigul', 'Erode', 'Kancheepuram', 'Krishnagiri',
       'Madurai', 'Nagercoil (Kannyiakumari)', 'Namakkal',
       'Ramanathapuram', 'Salem', 'Sivaganga', 'Thanjavur',
       'Thiruvannamalai', 'Vellore', 'Villupuram', 'Virudhunagar',
       'Hyderabad', 'Karimnagar', 'Khammam', 'Nalgonda', 'Nizamabad',
       'Ranga Reddy', 'North Tripura', 'Sepahijala', 'South District',
       'Agra', 'Aligarh', 'Allahabad', 'Baghpat', 'Bahraich', 'Ballia',
       'Bhadohi(Sant Ravi Nagar)', 'Bulandshahar', 'Chitrakut', 'Etah',
       'Etawah', 'Farukhabad', 'Fatehpur', 'Gautam Budh Nagar',
       'Ghaziabad', 'Jalaun (Orai)', 'Jhansi', 'Kannuj', 'Kanpur',
       'Khiri (Lakhimpur)', 'Lakhimpur', 'Mahoba', 'Mau(Maunathbhanjan)',
       'Mirzapur', 'Muradabad', 'Muzaffarnagar', 'Raebarelli', 'Rampur',
       'Saharanpur', 'Shahjahanpur', 'Sitapur', 'Sonbhadra', 'Dehradoon',
       'UdhamSinghNagar', 'Bankura', 'Burdwan', 'Hooghly', 'Jalpaiguri',
       'Medinipur(W)', 'Murshidabad', 'Nadia',
       'North 24 Parganas', 'Puruliya', 'Sounth 24 Parganas',
       'Uttar Dinajpur'))


        market_selection = st.sidebar.selectbox("Select Your Market", ('Port Blair', 'Kalikiri', 'Mulakalacheruvu', 'Vayalapadu',
       'Banaganapalli', 'Attili', 'Cachar', 'Kharupetia', 'Gauripur',
       'Jorhat', 'Pamohi(Garchuk)', 'Dhekiajuli', 'Jagdalpur', 'Charama',
       'Lakhanpuri', 'Narharpur', 'Pratappur', 'Sanquelim', 'Damnagar',
       'Khambhat(Grain Market)', 'Umreth', 'Ankleshwar', 'Bachau',
       'Nadiyad(Chaklasi)', 'Gogamba', 'Gogamba(Similiya)',
       'Bardoli(Katod)', 'Mandvi', 'S.Mandvi', 'Vyra', 'Bodeli',
       'Bodeli(Hadod)', 'Bodeli(Kalediya)', 'Bodeli(Modasar)', 'Padra',
       'Chikli(Khorgam)', 'Shahzadpur', 'Faridabad', 'Pataudi', 'Pipli',
       'Taura', 'Madlauda', 'Kangra(Jaisinghpur)', 'Kangra(Jassour)',
       'Palampur', 'Bhuntar', 'Zaloosa-Chararishrief (F&V)', 'Ramanagara',
       'Kolar', 'Gubbi', 'Aroor', 'Chengannur', 'Mannar', 'Piravam',
       'Taliparamba', 'Kasargod', 'Manjeswaram', 'Anchal', 'Chathanoor',
       'Kaliyanchanda', 'Ettumanoor', 'Kottayam', 'Kondotty',
       'Chelakkara', 'Irinjalakkuda', 'Kodungalloor', 'Maranelloor',
       'Parassala', 'Jaithari', 'Sendhwa', 'Dhar', 'Dhar(F&V)',
       'Gorakhpur', 'Thandla', 'Pandhana(F&V)', 'Gadarwada',
       'Sheopurkalan', 'Syopurkalan(F&V)', 'Newasa(Ghodegaon)', 'Rahata',
       'Deoulgaon Raja', 'Yawal', 'Kolhapur', 'Kolhapur(Malkapur)',
       'Savner', 'Bhokar', 'Lasalgaon(Niphad)', 'Pune', 'Pune(Hadapsar)',
       'Pune(Khadiki)', 'Pune(Pimpri)', 'Karad', 'Akluj', 'Mangal Wedha',
       'Pandharpur', 'Bishenpur', 'Moreh', 'Lamlong Bazaar', 'Imphal',
       'Thoubal', 'Shillong', 'Mangkolemba', 'Angul', 'Angul(Jarapada)',
       'Pallahara', 'Jaleswar', 'Nilagiri', 'Godabhaga', 'Hindol',
       'Kasinagar', 'Parlakhemundi', 'Bhanjanagar', 'Digapahandi',
       'Jharsuguda', 'Betnoti', 'Saraskana', 'Nawarangpur', 'Bonai',
       'Panposh', 'Karaikal', 'Chogawan', 'Mehta', 'Dhanaula', 'Bhucho',
       'Dera Baba Nanak', 'Dhariwal', 'Kalanaur', 'Dasuya', 'Adampur',
       'Lohian Khas', 'Mehatpur', 'Doraha', 'Sahnewal', 'Mansa', 'Malout',
       'Ghanaur', 'Rajpura', 'Morinda', 'Harike', 'Patti', 'Anta',
       'Barmer', 'Lunkaransar', 'Begu', 'Jalore', 'Khanpur', 'Itawa',
       'Sri Madhopur', 'Surajgarh', 'Deoli', 'Uniyara', 'Anaimalai',
       'Annur', 'Coimbatore', 'Karamadai', 'Madathukulam', 'Negamam',
       'Palladam', 'Pethappampatti', 'Pollachi', 'Pongalur',
       'Pudupalayam', 'Senjeri', 'Sevur', 'Thondamuthur', 'Udumalpet',
       'Cuddalore', 'Kurinchipadi', 'Dharampuri', 'Palakode',
       'Papparapatti', 'Pappireddipatti', 'Pennagaram', 'Gopalpatti',
       'Palani', 'Dharapuram', 'Elumathur', 'Kodumudi', 'Kunnathur',
       'Muthur', 'Vellakkoil', 'Acharapakkam', 'Chengalpattu',
       'Gummidipoondy', 'Kanchipuram', 'Maduranthagam', 'Pallipattu',
       'Ponner', 'Sunguvarchatram', 'Uthiramerur', 'Bargur',
       'Pochampalli', 'Uthangarai', 'Melur', 'Thirumangalam',
       'Kalliakavillai', 'Namagiripettai', 'Namakkal', 'Rasipuram',
       'Tiruchengode', 'Velur', 'Sivagangai', 'Attur', 'Gangavalli',
       'Karumanturai', 'Kolathur', 'Konganapuram', 'Omalur', 'Salem',
       'Thalaivasal', 'Thammampati', 'Vazhapadi', 'Singampuneri',
       'Budalur', 'Kumbakonam', 'Orathanadu', 'Papanasam', 'Thanjavur',
       'Thiruppananthal', 'Vallam', 'Cheyyar', 'Ammoor', 'Kalavai',
       'Vellore', 'Avalurpet', 'Chinnasalem', 'Gingee', 'Kallakurichi',
       'Manalurpet', 'Sangarapuram', 'Tindivanam', 'Tiruvennainallur',
       'Vikkiravandi', 'Villupuram', 'Rajapalayam', 'Sathur',
       'Mahboob Manison', 'Koratla', 'Vemulawada', 'Madhira', 'Nalgonda',
       'Ramannapet', 'Voligonda', 'Pitlam', 'Chevella', 'Dasda',
       'Bishalgarh', 'Bishramganj', 'Barpathari', 'Achnera', 'Fatehabad',
       'Jagnair', 'Jarar', 'Khairagarh', 'Samsabad', 'Atrauli', 'Sirsa',
       'Bagpat', 'Baraut', 'Mihipurwa', 'Risia', 'Ruperdeeha',
       'Chitwadagaon', 'Gopiganj', 'Divai', 'Gulavati', 'Siyana',
       'Mau(Chitrakut)', 'Aliganj', 'Jasvantnagar', 'Kamlaganj', 'Bindki',
       'Kishunpur', 'Javer', 'Hapur', 'Ait', 'Kadaura', 'Baruwasagar',
       'Chirgaon', 'Chhibramau(Kannuj)', 'Rura', 'Mohammdi', 'Tikonia',
       'Paliakala', 'Charkhari', 'Doharighat', 'Mirzapur', 'Sambhal',
       'Muzzafarnagar', 'Shahpur', 'Thanabhawan', 'Lalganj', 'Milak',
       'Chutmalpur', 'Devband', 'Gangoh', 'Nakud', 'Nanuta',
       'Sultanpurchilkana', 'Jalalabad', 'Viswan', 'Dudhi', 'Robertsganj',
       'Dehradoon', 'Kashipur', 'Bishnupur(Bankura)', 'Khatra', 'Asansol',
       'Burdwan', 'Guskara(Burdwan)', 'Kalna', 'Katwa', 'Memari',
       'Pandua', 'Dhupguri', 'Samsi', 'Ghatal', 'Beldanga', 'Jangipur',
       'Nadia', 'Habra', 'Balarampur', 'Purulia', 'Baruipur(Canning)',
       'Islampur', 'Raiganj'))


        commodity_selection = st.sidebar.selectbox("Select your Commodity",('Amaranthus', 'Banana - Green', 'Bhindi(Ladies Finger)',
       'Bitter gourd', 'Black pepper', 'Bottle gourd', 'Brinjal',
       'Cabbage', 'Carrot', 'Cauliflower', 'Cluster beans', 'Coconut',
       'Colacasia', 'Onion', 'Potato', 'Tomato',
       'Bengal Gram(Gram)(Whole)', 'Jowar(Sorghum)',
       'Paddy(Dhan)(Common)', 'Lentil (Masur)(Whole)', 'Rice',
       'Cucumbar(Kheera)', 'Field Pea', 'French Beans (Frasbean)',
       'Green Chilli', 'Knool Khol', 'Pumpkin', 'Raddish',
       'Black Gram (Urd Beans)(Whole)', 'Green Gram (Moong)(Whole)',
       'Jute', 'Maida Atta', 'Mustard', 'Wheat Atta', 'Garlic',
       'Masur Dal', 'Ridgeguard(Tori)', 'Arecanut(Betelnut/Supari)',
       'Arhar (Tur/Red Gram)(Whole)', 'Maize', 'Dry Chillies',
       'Groundnut', 'Capsicum', 'Guar', 'Lemon',
       'Bajra(Pearl Millet/Cumbu)', 'Castor Seed', 'Coriander(Leaves)',
       'Cowpea(Veg)', 'Drumstick', 'Elephant Yam (Suran)',
       'Ginger(Green)', 'Indian Beans (Seam)', 'Methi(Leaves)',
       'Onion Green', 'Peas cod', 'Pegeon Pea (Arhar Fali)',
       'Sponge gourd', 'Surat Beans (Papadi)', 'Sweet Potato', 'Tinda',
       'Guar Seed(Cluster Beans Seed)', 'Cotton', 'Wheat',
       'Gram Raw(Chholia)', 'Little gourd (Kundru)', 'Round gourd',
       'Leafy Vegetable', 'Mint(Pudina)', 'Papaya (Raw)', 'Spinach',
       'Pointed gourd (Parval)', 'Banana', 'Ber(Zizyphus/Borehannu)',
       'Grapes', 'Kinnow', 'Peas Wet', 'Apple', 'Orange', 'Pomegranate',
       'Papaya', 'Chikoos(Sapota)', 'Mashrooms', 'Mousambi(Sweet Lime)',
       'Pineapple', 'Guava', 'Turnip', 'Ginger(Dry)',
       'Squash(Chappal Kadoo)', 'Beans', 'Beetroot', 'Chilly Capsicum',
       'Green Avare (W)', 'Seemebadnekai', 'Snakeguard', 'Suvarna Gadde',
       'Water Melon', 'Copra', 'Amphophalus', 'Ashgourd', 'Coconut Oil',
       'Rubber', 'Cashewnuts', 'Pepper garbled', 'Coconut Seed',
       'Long Melon(Kakri)', 'Tapioca', 'Turmeric (raw)',
       'Mango (Raw-Ripe)', 'Amla(Nelli Kai)', 'Duster Beans', 'Mango',
       'Soyabean', 'Linseed', 'Niger Seed (Ramtil)',
       'Green Gram Dal (Moong Dal)', 'Lime', 'Turmeric',
       'Karbuja(Musk Melon)', 'Pear(Marasebu)', 'Rajgir', 'Sweet Pumpkin',
       'Tender Coconut', 'Bengal Gram Dal (Chana Dal)', 'Betal Leaves',
       'Broken Rice', 'Gur(Jaggery)', 'Sugar',
       'Sesamum(Sesame,Gingelly,Til)', 'Moath Dal', 'Corriander seed',
       'Ground Nut Seed', 'Taramira', 'Tobacco', 'Tamarind Fruit',
       'Cowpea (Lobia/Karamani)', 'Kulthi(Horse Gram)',
       'Ragi (Finger Millet)', 'T.V. Cumbu', 'Gingelly Oil',
       'Kodo Millet(Varagu)', 'Hybrid Cumbu', 'Karamani',
       'Thinai (Italian Millet)', 'Wood', 'Barley (Jau)', 'Fish',
       'Green Peas', 'Arhar Dal(Tur Dal)', 'Black Gram Dal (Urd Dal)',
       'Mustard Oil', 'Ghee', 'Paddy(Dhan)(Basmati)', 'White Pumpkin',
       'Peas(Dry)', 'Plum'))


        variety_selection = st.sidebar.selectbox("Select the variety",('Amaranthus', 'Banana - Green', 'Bhindi', 'Other', 'Cluster Beans',
       'Big', 'Local', 'Desi (Whole)', 'Jowar ( White)', 'Sona Mahsuri',
       'Paddy', 'Masur Dal', 'Common', 'Fine', 'Bitter Gourd',
       'Bottle Gourd', 'Round', 'Cabbage', 'Carrot', 'Cucumbar',
       'Field Pea', 'French Beans (Frasbean)', 'Green Chilly',
       'Knool Khol', 'Badshah', 'Pumpkin', 'Raddish', 'Hybrid',
       'Black Gram (Whole)', 'Green (Whole)', 'TD-5', 'Maida Atta',
       'Mustard', 'Medium', 'Wheat Atta', 'Garlic', 'Onion', 'Potato',
       'Brinjal', 'Cauliflower', 'Ridgeguard(Tori)', 'Tomato', 'Raw',
       'Dry', 'Red', 'Supari', 'White', 'Capsicum', 'Gwar', 'Lemon',
       'Deshi', 'Castor seed', 'Coriander', 'Cowpea (Veg)', 'Drumstick',
       'Elephant Yam (Suran)', 'Green Ginger', 'Indian Beans (Seam)',
       'Methi', 'Onion Green', 'Peas cod', 'Pigeon Pea (Arhar Fali)',
       'Sponge gourd', 'Surat Beans (Papadi)', 'Sweet Potato', 'Tinda',
       'Whole', 'Deshi White', 'G. R. 11', 'Sabnam',
       'Shanker 6 (B) 30mm FIne', 'Leafy Vegetables', 'Ber(Zizyphus)',
       'Desi', 'Green', 'Peas Wet', 'Spinach', 'Kinnow', 'Apple',
       'Beans (Whole)', 'Beetroot', 'Chilly Capsicum', 'Green Avare (W)',
       'Papaya', 'Seemebadanekai', 'Snakeguard', 'Suvarnagadde',
       'Water Melon', 'other', 'Amphophalus', 'Nendra Bale',
       'Palayamthodan', 'Robusta', 'Ashgourd', 'Copra', 'Poovan',
       'Red Banana', 'Banana - Ripe', 'Mango - Raw-Ripe', 'Tapioca',
       'Rasakathai', 'Round/Long', 'Colacasia', 'Yellow', 'Lokwan',
       'DCH-32  (Ginned)', 'Green Gram Dal', '1001', '147 Average',
       'Hapus(Alphaso)', 'Simla', 'Hosur Red', 'Bengal Gram Dal',
       'Turmeric (raw)', 'New Variety', 'Champa', 'Orange', 'api',
       'Sanna Bhatta', 'Broken Rice', 'NO 2', 'Hybrid/Local', 'B P T',
       'Pommani', 'White Ponni', 'Sesame', 'American', 'Black', 'Nasik',
       'Sapota', 'Guava', 'Pine Apple', 'Pomogranate',
       'Squash(Chappal Kadoo)', '999', 'Bulb', 'Coconut', 'Ball', 'DMV-7',
       'Hybrid Red (Cattle Feed)', 'Chapathi', 'ADT 37', 'ADT 43',
       'ADT 36', 'MCU 5', 'Ponni', 'Local(Raw)', 'Finger', 'GCH',
       'Super Ponni', 'RCH-2', 'TKM 9', '95/5', 'A. Ponni', 'ADT 39',
       'Bold Kernel', '1st Sort', '2nd Sort', 'Cotton (Unginned)',
       ' Subabul', 'MTU-1010', 'African Sarson', 'Kala Masoor New',
       'Masuri', 'Papaya (Raw)', 'Jalander', 'Dara', 'Sarson(Black)',
       'III', 'Delicious', 'Pathari', 'Mousambi', 'Kasmir/Shimla - II',
       'Arkasheela Mattigulla', '(Red Nanital)', 'Chips', 'Paddy Coarse',
       'Rahu(Local)', 'Laha(Sarsib)', 'F.A.Q.', '777 New Ind',
       'Big 100 Kg', '1009 Kar', 'Beedi', 'Green Peas', 'Arhar Dal(Tur)',
       'Black Gram Dal', 'Lohi Black', 'Mustard Oil', 'Ghee', 'Ankola',
       'Annigeri', 'Hybrid Yellow (Cattle Feed)', '1121', 'Beete (Rose)',
       'Khandsari', 'White Pumpkin', 'Annabesahai', 'Nagpuri', 'Coarse',
       'Chini', 'Masoor Gola', 'Peas(Dry)', 'Plum', 'Disara', 'Jyoti',
       'Sonalika', 'Swarna Masuri (New)', 'Sweet Pumpkin',
       'Green Gram Dhall-I', 'Kalyan', 'Yellow (Black)', 'Ratna',
       'Ratnachudi (718 5-749)', 'H.Y.V.', 'Moath Dal', 'Beelary-Red',
       'Super Fine'))

        result=""
        if st.button("Predict"):
            result=crop_price_prediction([state_selection, district_selection, market_selection, commodity_selection, variety_selection])


        st.success('The suggested crop price (per quintal) is {}'.format(result))


    elif selection == 'Yield Prediction':
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Yield Prediction</b></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(textbg.format(bgcolor,fontcolor),unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b>Forecasting or predicting the crop yield well ahead of its harvest time would assist the strategists and farmers for taking suitable measures for selling and storage. In addition to such human errors, the fluctuations in the prices themselves make creating a stable and robust forecasting solution a necessity.</b></font></h1>
        </div>
        """
        bgcolor = ''
        fontcolor = 'white'
        st.markdown(text.format(bgcolor,fontcolor),unsafe_allow_html=True)

        st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
         unsafe_allow_html=True,)

        # Updated dropdown for state selection using data from state_district_data.py
        state_selection = st.sidebar.selectbox("Select your state", states)

        # Updated dropdown for district selection based on selected state
        if state_selection in districts:
            district_selection = st.sidebar.selectbox("Select your district", districts[state_selection])
        else:
            district_selection = st.sidebar.selectbox("Select your district", ["No districts available"])

        # Crop year selection
        crop_year = st.sidebar.slider("Select crop year", 2025, 2026, 2040)

        # Updated dropdown for season selection  
        season_select = st.sidebar.selectbox("Select the season", seasons)
        
        # Format season with proper spacing as needed by model
        if season_select == 'kharif':
            season_select = 'Kharif     '
        elif season_select == 'whole year':
            season_select = 'Whole Year '
        elif season_select == 'autumn':
            season_select = 'Autumn     '
        elif season_select == 'rabi':
            season_select = 'Rabi       '
        elif season_select == 'summer':
            season_select = 'Summer     '
        else:
            season_select = 'Winter     '

        # Updated dropdown for crop selection
        crop_selection = st.sidebar.selectbox("Select your crop", crops)

        # Area input field
        area_selection = st.sidebar.text_input("Enter the area of your field (in hectares) - 1 hectare = 10,000 square meters")

        result=""
        if st.button("Predict"):
            result=yield_prediction([state_selection, district_selection, crop_year, season_select, crop_selection, area_selection])

        if result:
            st.success('The estimated crop yield is {:.2f} TON per hectare'.format(float(result)))
        elif result == 0:
            st.success('The estimated crop yield is 0.00 kg per hectare')
        else:
            st.error('Submit your input values.')





        


            




if __name__ == '__main__':
    main()





