import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle

# Load the dataset
df = pd.read_csv("updated_dataset_aj.csv")

# Handle missing values
df.fillna('', inplace=True)

# One-hot encode categorical features
ohe_specialization = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ohe_disease_type = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

specialization_encoded = ohe_specialization.fit_transform(df[['Specialization']])
disease_type_encoded = ohe_disease_type.fit_transform(df[['Disease_Type']])

specialization_cols = ohe_specialization.get_feature_names_out(['Specialization'])
disease_type_cols = ohe_disease_type.get_feature_names_out(['Disease_Type'])

df_encoded = pd.DataFrame(specialization_encoded, columns=specialization_cols)
df_encoded[disease_type_cols] = pd.DataFrame(disease_type_encoded, index=df_encoded.index)

# Normalize numerical features
scaler = StandardScaler()
df[['Experience(Years)', 'Total_Reviews', 'Patient Satisfaction Rate(%age)']] = scaler.fit_transform(
    df[['Experience(Years)', 'Total_Reviews', 'Patient Satisfaction Rate(%age)']]
)

# Merge encoded features with original DataFrame
df_merged = pd.concat([df, df_encoded], axis=1)

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_merged['Cluster'] = kmeans.fit_predict(df_encoded)

# Save the models and encoders
pickle.dump(kmeans, open('kmeans.pkl', 'wb'))
pickle.dump(ohe_specialization, open('ohe_specialization.pkl', 'wb'))
pickle.dump(ohe_disease_type, open('ohe_disease_type.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Recommendation functions
def recommend_doctors_by_specialization(specialization):
    # Use the same OneHotEncoder used during training
    specialization_encoded = ohe_specialization.transform([[specialization]])

    # Ensure that the encoded specialization matches the number of features expected by KMeans
    specialization_encoded = pd.DataFrame(specialization_encoded, columns=specialization_cols).reindex(columns=df_encoded.columns, fill_value=0)

    # Predict the cluster
    cluster = kmeans.predict(specialization_encoded)[0]

    # Filter doctors based on the predicted cluster
    doctors = df_merged[df_merged['Cluster'] == cluster]

    # Apply ranking logic
    doctors = filter_by_rank(doctors)

    # Return top 10 doctors
    top_doctors = doctors.head(10)
    return top_doctors

def filter_by_disease_type(doctors, disease_type):
    return doctors[doctors['Disease_Type'] == disease_type]

def filter_by_city(doctors, city):
    return doctors[doctors['City'].str.contains(city, case=False, na=False)]

def filter_by_experience(doctors, min_experience):
    return doctors[doctors['Experience(Years)'] >= min_experience]

def filter_by_reviews(doctors, min_reviews):
    return doctors[doctors['Total_Reviews'] >= min_reviews]

def filter_by_rank(doctors):
    doctors['Rank'] = doctors['Experience(Years)'] * 0.4 + doctors['Total_Reviews'] * 0.3 + doctors['Patient Satisfaction Rate(%age)'] * 0.3
    return doctors.sort_values(by='Rank', ascending=False)

def get_additional_fields(doctors, fields):
    return doctors[fields]