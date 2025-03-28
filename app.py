import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.preprocessing import OrdinalEncoder

# Load the ML model and encoders
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Set background image
def set_background_image_local(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as file:
            data = file.read()
        base64_image = base64.b64encode(data).decode("utf-8")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{base64_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}     
            </style>
            """,
            unsafe_allow_html=True
        )

set_background_image_local("bugatti-chiron-super-sport-wallpaper-3840x2160_54.jpg")

# Load models
try:
    model_car = load_model("MLmodel_name_1.pkl")
    encoder_city = load_model("encoder_city.pkl")
    encoder_Insurance_Validity = load_model("encoder_Insurance_Validity.pkl")
    encoder_bt = load_model("encoder_bt.pkl")
    encoder_ft = load_model("encoder_ft.pkl")
    encoder_oem = load_model("encoder_oem.pkl")
    encoder_model = load_model("encoder_model_car.pkl")
    encoder_transmission = load_model("encoder_transmission.pkl")
    encoder_variantName = load_model("encoder_variantName.pkl")
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please check if all encoder files and model are present.")

# Load dataset
try:
    cd_df = pd.read_csv("FINALDATA1 (1).csv")
except FileNotFoundError:
    st.error("Error: The dataset file 'FINALDATA1 (1).csv' is missing!")

st.title("Car Price Prediction App")

# Extract unique values for dropdowns
categorical_features = ["city", "ft", "bt", "transmission", "oem", "model", "variantName", "Insurance Validity"]
dropdown_options = {feature: cd_df[feature].unique().tolist() for feature in categorical_features}

# Tabs
tab1, tab2 = st.tabs(['Home', 'Chat-Bot'])

### TAB 1: Car Price Prediction
with tab1:
    # Create UI layout
    a1, a2, a3 = st.columns(3)
    a4, a5, a6 = st.columns(3)
    a7, a8, a9 = st.columns(3)
    a10, a11, a12 = st.columns(3)
    a13, a14 = st.columns(2)
    
    with a1:
        city_select = st.selectbox("Select City", dropdown_options["city"])
        city = encoder_city.transform([[city_select]])[0][0]
    
    with a2:
        ft_select = st.selectbox("Select Fuel Type", dropdown_options["ft"])
        ft = encoder_ft.transform([[ft_select]])[0][0]
    
    with a3:
        bt_select = st.selectbox("Select Body Type", dropdown_options["bt"])
        bt = encoder_bt.transform([[bt_select]])[0][0]
    
    with a4:
        km = st.number_input("Enter KM driven", min_value=10)
    
    with a5:
        transmission_select = st.selectbox("Select Transmission", dropdown_options["transmission"])
        transmission = encoder_transmission.transform([[transmission_select]])[0][0]
    
    with a6:
        ownerNo = st.number_input("Enter No. of Owners", min_value=1)
    
    with a7:
        oem_filtered = cd_df.loc[cd_df["ft"] == ft_select, "oem"].unique().tolist()
        oem_select = st.selectbox("Select Car Manufacturer", oem_filtered)
        oem = encoder_oem.transform([[oem_select]])[0][0]
    
    with a8:
        model_filtered = cd_df.loc[cd_df["oem"] == oem_select, "model"].unique().tolist()
        model_select = st.selectbox("Select Car Model", model_filtered)
        model = encoder_model.transform([[model_select]])[0][0]
    
    with a9:
        modelYear = st.number_input("Enter Manufacture Year", min_value=1900)
    
    with a10:
        variantName_filtered = cd_df.loc[cd_df["model"] == model_select, "variantName"].unique().tolist()
        variantName_select = st.selectbox("Select Variant Name", variantName_filtered)
        variantName = encoder_variantName.transform([[variantName_select]])[0][0]
    
    with a11:
        Registration_Year = st.number_input("Enter Registration Year", min_value=1900)
    
    with a12:
        InsuranceValidity_select = st.selectbox("Select Insurance Validity", dropdown_options["Insurance Validity"])
        InsuranceValidity = encoder_Insurance_Validity.transform([[InsuranceValidity_select]])[0][0]
    
    with a13:
        Seats = st.number_input("Enter Seat Capacity", min_value=4)
    
    with a14:
        EngineDisplacement = st.number_input("Enter Engine CC", min_value=799)

    # Prediction Button
    if st.button('Predict'):
        input_features = [
            city, ft, bt, km, transmission, ownerNo, oem, model, 
            modelYear, variantName, Registration_Year, InsuranceValidity, Seats, EngineDisplacement
        ]
        input_array = np.array(input_features).reshape(1, -1)
        prediction = model_car.predict(input_array)
        
        st.subheader("Predicted Car Price")
        st.markdown(f"### :green[₹ {prediction[0]:,.2f}]")

### TAB 2: Chatbot
with tab2:
    st.write("Welcome to the Car Dealership Chatbot! ")

    # Define the chatbot's behavior
    car_dealership_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant at a car dealership. "
                         "You can provide information about car models, pricing, and financing options. "
                         "Be friendly and informative."),
            ("placeholder", "{messages}"),
        ]
    )

    # Instantiate the chat model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="AIzaSyBpufebS9kziyw2coPefRL0wtXDs5wKbjM")

    # Streamlit UI
    st.title("🚗 Car Dealership Chatbot")
    st.write("Ask me anything about car models, pricing, and financing options!")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style="color: green; font-weight: bold; font-size: 16px;">
                        {message["content"]}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask me a question...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
    
        with st.chat_message("user"):
            st.markdown(user_input)
    
        # Get response from chatbot
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["messages"]]
        response = llm.invoke(messages)
    
        with st.chat_message("assistant"):
            st.markdown(
                f"""
                <div style="color: green; font-weight: bold; font-size: 16px;">
                    {response.content}
                </div>
                """, 
                unsafe_allow_html=True
            )
    
        # Store assistant response
        st.session_state["messages"].append({"role": "assistant", "content": response.content})
