import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

# Import recommendation functions from model.py
from model8 import recommend_doctors_by_specialization, filter_by_disease_type, filter_by_city, filter_by_experience, filter_by_reviews, filter_by_rank, get_additional_fields

st.title('Welcome to the Doctor Recommender System')

# Load pre-trained models and encoders
kmeans = pickle.load(open('kmeans.pkl', 'rb'))
ohe_specialization = pickle.load(open('ohe_specialization.pkl', 'rb'))
ohe_disease_type = pickle.load(open('ohe_disease_type.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the dataset
df = pd.read_csv('updated_dataset_aj.csv')

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a section:', ['Home', 'Recommendations', 'Feedback', 'Dashboard & Analytics'])

# Home section
if options == 'Home':
    st.write("Use the sidebar to navigate to different sections.")

# Recommendations section
if options == 'Recommendations':
    st.title('Doctor Recommendations')

    specialization = st.selectbox('Select Specialization', df['Specialization'].unique())
    disease_type = st.selectbox('Select Disease Type', df['Disease_Type'].unique())
    city = st.text_input('City')
    min_experience = st.slider('Minimum Experience (Years)', 0, 50, 0)
    min_reviews = st.slider('Minimum Reviews', 0, 1000, 0)

    # Additional fields to display
    additional_fields = st.multiselect('Additional Fields to Display', ['Doctor Qualification', 'Experience(Years)', 'Total_Reviews', 'Patient Satisfaction Rate(%age)', 'City'])

    # Buttons to apply filters
    use_specialization = st.checkbox('Use Specialization Filter', value=True)
    use_disease_type = st.checkbox('Use Disease Type Filter')
    use_city = st.checkbox('Use City Filter')
    use_experience = st.checkbox('Use Experience Filter')
    use_reviews = st.checkbox('Use Reviews Filter')

    if st.button('Get Recommendations'):
        doctors = recommend_doctors_by_specialization(specialization)
        
        if use_disease_type:
            doctors = filter_by_disease_type(doctors, disease_type)
        if use_city:
            doctors = filter_by_city(doctors, city)
        if use_experience:
            doctors = filter_by_experience(doctors, min_experience)
        if use_reviews:
            doctors = filter_by_reviews(doctors, min_reviews)
        
        doctors = filter_by_rank(doctors)
        
        if additional_fields:
            doctors = get_additional_fields(doctors, ['Doctor Name', 'Specialization', 'City'] + additional_fields)
        else:
            doctors = doctors[['Doctor Name', 'Specialization', 'City', 'Experience(Years)']]

        # Remove the index and format the experience column
        doctors.reset_index(drop=True, inplace=True)
        doctors['Experience(Years)'] = doctors['Experience(Years)'].apply(lambda x: f"{int(x)} years")

        st.write(doctors)

# Feedback section
if options == 'Feedback':
    st.title('Doctor Feedback')
    
    doctor_name = st.text_input('Doctor Name')
    rating = st.slider('Rate the Doctor (1-5)', 1, 5)
    comments = st.text_area('Comments')

    if st.button('Submit Feedback'):
        feedback_data = {
            'Doctor Name': doctor_name,
            'Rating': rating,
            'Comments': comments
        }
        feedback_df = pd.DataFrame([feedback_data])
        feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
        st.success('Feedback Submitted')

# Dashboard & Analytics section
if options == 'Dashboard & Analytics':
    st.title('Dashboard and Analytics')

    # Popular Specializations
    st.subheader('Popular Specializations')
    specialization_counts = df['Specialization'].value_counts()
    st.bar_chart(specialization_counts)

    # Highest Rated Doctors
    st.subheader('Highest Rated Doctors')
    top_doctors = df.nlargest(10, 'Patient Satisfaction Rate(%age)')
    fig = px.bar(top_doctors, x='Doctor Name', y='Patient Satisfaction Rate(%age)', color='Specialization')
    st.plotly_chart(fig)

    # Experience Distribution
    st.subheader('Experience Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['Experience(Years)'].dropna(), kde=True, ax=ax)
    st.pyplot(fig)
