import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a dictionary of agencies and their corresponding values

# Define mappings from cross-reference file

CLUSTER_DESCRIPTIONS = {
    0: {
        "title": "High-Visit Moderate-Income Families",
        "avg_age": 49.69,
        "gender": "Female",
        "ethnicity": "Hispanic/Latino, White/Anglo",
        "disability": 2.53,
        "household_size": 4.12,
        "income": "$1,370.73",
        "visits": 7.93,
        "key_obs": "Moderate income, high visits, larger household size"
    },
    1: {
        "title": "Low-Income Elderly Group",
        "avg_age": 67.84,
        "gender": "Female",
        "ethnicity": "Middle-Eastern, White/Anglo",
        "disability": 0.99,
        "household_size": 1.79,
        "income": "$1,035.17",
        "visits": 1.96,
        "key_obs": "Older group, smaller households, lower income, fewer visits"
    },
    2: {
        "title": "Moderate-Income Middle-Aged Males",
        "avg_age": 54.42,
        "gender": "Male",
        "ethnicity": "White/Anglo, Other",
        "disability": 2.95,
        "household_size": 2.53,
        "income": "$1,175.48",
        "visits": 2.12,
        "key_obs": "Moderate age, moderate income, higher disability rate, moderate visits"
    },
    3: {
        "title": "Younger Moderate-Income Families",
        "avg_age": 40.29,
        "gender": "Female",
        "ethnicity": "Middle-Eastern, White/Anglo",
        "disability": 1.29,
        "household_size": 4.32,
        "income": "$1,203.24",
        "visits": 2.1,
        "key_obs": "Younger group, moderate income, larger households, moderate visits"
    },
    4: {
        "title": "High-Income Large Families",
        "avg_age": 47.48,
        "gender": "Female",
        "ethnicity": "Hispanic/Latino, White/Anglo",
        "disability": 0.99,
        "household_size": 5.89,
        "income": "$39,631.19",
        "visits": 3.03,
        "key_obs": "Exceptionally high income, large household size, moderate visits"
    },
    5: {
        "title": "Low-Income Older Males",
        "avg_age": 56.04,
        "gender": "Male",
        "ethnicity": "Middle-Eastern, White/Anglo",
        "disability": 0.88,
        "household_size": 2.4,
        "income": "$1,043.59",
        "visits": 2.05,
        "key_obs": "Older, lower income, small households"
    },
    6: {
        "title": "Younger Moderate Income Females",
        "avg_age": 40.36,
        "gender": "Female",
        "ethnicity": "Middle-Eastern, White/Anglo",
        "disability": 0.97,
        "household_size": 4.46,
        "income": "$1,313.39",
        "visits": 2.14,
        "key_obs": "Younger, larger households, moderate income"
    },
    7: {
        "title": "Moderate-Income Middle-Aged Females",
        "avg_age": 53.36,
        "gender": "Female",
        "ethnicity": "White/Anglo, Other",
        "disability": 3.11,
        "household_size": 2.78,
        "income": "$1,306.31",
        "visits": 2.14,
        "key_obs": "Moderate age, moderate income, higher disability"
    }
}

GENDER_MAPPING = {
    "Didn't Ask": 0,
    "Female": 1,
    "Femalefemale": 2,
    "Male": 3,
    "Malemale": 4,
    "None of These": 5,
    "Prefer Not to Answer": 6,
    "Transgender": 7
}

ETHNICITY_MAPPING = {
    "White / Anglo": 37,
    "Black / African American": 19,
    "Asian": 12,
    "Hispanic / Latino": 26,
    "Middle-Eastern / North-African": 30,
    "Pacific Islander": 34,
    "American Indian / Native American": 4,
    "Alaska Native / Aleut / Eskimo": 0,
    "Unknown": 36,
    "declined_to_answer": 39,
    "did_not_ask": 40,
    "do_not_know": 41
}

DISABILITY_MAPPING = {
    "Unknown": 0,
    "did_not_ask": 1,
    "do_not_know": 2,
    "no": 3,
    "prefer_not_to_answer": 4,
    "yes": 5
}

SELF_IDENTIFIES_MAPPING = {
    "Active Military": 0,
    "At Risk Of Being Homeless": 1,
    "Disability": 3,
    "Unknown": 5,
    "declined_to_answer": 6,
    "did_not_ask": 7,
    "do_not_know": 9,
    "none": 10,
    "other": 11,
    "veteran": 13
}
AGENCIES = {
    "10-7 Farms": 0,
    "Agape Fellowship Ministries": 1,
    "Antioch United Methodist Church": 2,
    "Asbury Manor": 3,
    "Barbara Carroll Community Outreach & Development": 4,
    "Bowling Green Properties": 5,
    "Chancellor Baptist Church": 6,
    "Christ Episcopal Church": 7,
    "Christian Brothers Transitional Program": 8,
    "Community Ministry Center": 9,
    "Concord Baptist Church": 10,
    "Door Dash": 11,
    "Eastland United Methodist Church": 12,
    "Emmanuel AME Church": 13,
    "FRFB's Mobile Distributions": 14,
    "Fairview Baptist Church": 15,
    "Fawn Lake Mens Christian Fellowship": 16,
    "Forest Village Apartments": 17,
    "Fredericksburg Regional Food Bank": 18,
    "Fredericksburg United Methodist Church": 19,
    "Garden of Delight/Iglesia Jardin de Delicias": 20,
    "Garrison Woods Apartments": 21,
    "God's Holy Temple": 22,
    "Hartwood Presbyterian Church": 23,
    "Hazel Hill Apartments": 24,
    "Healthy Generations Area Agency on Aging": 25,
    "Here 2 Serve": 26,
    "Heritage Park": 27,
    "Hideaway Townhomes": 28,
    "Highway Assembly of God": 29,
    "Hollywood Church of the Brethren": 30,
    "Humanities Foundation - New Post Site": 31,
    "Humanities Foundation, Inc.": 32,
    "Humanities Foundation- Keswick Senior Apartments (CSFP)": 33,
    "Islamic Center of Fredericksburg": 34,
    "King George Church of God": 35,
    "Kings Crest Senior Community": 36,
    "Knights of Columbus": 37,
    "Lions Wilderness Food Pantry": 38,
    "Little Ark Baptist Church": 39,
    "Love Thy Neighbor": 40,
    "Lucha Ministries, Inc.": 41,
    "Madonna House": 42,
    "Massaponax Baptist Church": 43,
    "Micah Ecumenical Ministries": 44,
    "Mill Park Terrace Apartments": 45,
    "Mobile Pantry @ Beulah Baptist Church": 46,
    "Mobile Pantry @ Bowling Green Baptist Church": 47,
    "Mobile Pantry @ Caroline Community Services Center": 48,
    "Mobile Pantry @ Dahlgren Harbor Apartments": 49,
    "Mobile Pantry @ Encounter Church of God": 50,
    "Mobile Pantry @ First Baptist Ambar Church": 51,
    "Mobile Pantry @ Germanna Community College": 52,
    "Mobile Pantry @ Hazel Hill Apartments": 53,
    "Mobile Pantry @ Heritage Park": 54,
    "Mobile Pantry @ King George DSS": 55,
    "Mobile Pantry @ Livingston Elementary School": 56,
    "Mobile Pantry @ Lloyd F. Moss Free Clinic": 57,
    "Mobile Pantry @ Lotus Academy": 58,
    "Mobile Pantry @ Massaponax High School": 59,
    "Mobile Pantry @ Meadow Event Park": 60,
    "Mobile Pantry @ New Liberty Baptist Church": 61,
    "Mobile Pantry @ New Post Apts": 62,
    "Mobile Pantry @ North Stafford Church of Christ": 63,
    "Mobile Pantry @ Partlow Ruritan Building": 64,
    "Mobile Pantry @ Port Royal Library": 65,
    "Mobile Pantry @ R&D Family Campground": 66,
    "Mobile Pantry @ Rappahannock Goodwill": 67,
    "Mobile Pantry @ Second Mt. Zion Baptist Church": 68,
    "Mobile Pantry @ Shirley Heim Middle School": 69,
    "Mobile Pantry @ Spotswood Elementary School": 70,
    "Mobile Pantry @ Strong Tower Church": 71,
    "Mobile Pantry @ The Garden Inn (Travelodge)": 72,
    "Mobile Pantry @ Third Mt. Zion Baptist Church": 73,
    "Mobile Pantry @ Widewater Elementary School": 74,
    "Mt. Zion Baptist Church (Triangle)": 75,
    "New Hope Baptist Church": 76,
    "New Hope Christian Ministries": 77,
    "New Vision Ministries": 78,
    "Oak Grove Baptist Church": 79,
    "Peace United Methodist Church": 80,
    "Praise Temple Apostolic Faith Church of Virginia": 81,
    "Ramoth Baptist Church": 82,
    "Real Life Community Church": 83,
    "Rehoboth United Methodist Church": 84,
    "Saint Mary of the Annunciation": 85,
    "Shiloh Baptist Church (Old Site)": 86,
    "Spotswood Baptist Church": 87,
    "Spotsylvania Emergency Concerns Association (SECA)": 88,
    "St. Anthony of Padua Catholic Church": 89,
    "St. George's Episcopal Church - The Table": 90,
    "St. Matthew Catholic Church": 91,
    "St. Matthias United Methodist Church": 92,
    "St. Peter's Lutheran Church": 93,
    "Stafford Church of God": 94,
    "Stafford Emergency Relief through Volunteer Effort (SERVE)": 95,
    "Sylvania Heights Baptist Church": 96,
    "The Gardens of Stafford": 97,
    "The Salvation Army": 98,
    "Triangle Baptist Church": 99,
    "Trinity Episcopal Church": 100,
    "United Faith Christian Ministry": 101,
    "Wilderness Community Church": 102,
    "Wright's Chapel United Methodists": 103,
    "Zion Church of Fredericksburg": 104,
    "Zion Hill Baptist Church": 105,
    "Zion United Methodist Church": 106
}



@st.cache_resource
def load_models():
    try:
        kmeans_model = joblib.load("kmeans.pkl")
        decision_tree_model = joblib.load("best_tree_model.pkl")
        st.success("Models loaded successfully!")
        return kmeans_model, decision_tree_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def preprocess_data(input_dict):
    """Preprocess the input data for model prediction"""
    # Create base DataFrame from input
    df = pd.DataFrame([input_dict])
    
    # Ensure correct column order for clustering - using prev_visits for Avg Visits Per Month
    clustering_columns = [
        'Client Age', 
        'Client Gender Identity-Labels',
        'Client Ethnicity-Labels',
        'Client Disability',
        'Client Self-Identifies As',
        'Household Size',
        'Monthly Household Income',
        'Visited Agency',
        'Avg Visits Per Month'  # This will come from prev_visits
    ]
    clustering_data = df.copy()
    clustering_data = clustering_data[clustering_columns]
    
    # Ensure correct column order for decision tree
    dt_columns = [
        'Client Age',
        'Client Gender Identity-Labels',
        'Client Ethnicity-Labels',
        'Client Disability',
        'Client Self-Identifies As',
        'Household Size',
        'Monthly Household Income',
        'Visited Agency'
    ]
    decision_tree_data = df[dt_columns]
    
    return clustering_data, decision_tree_data

def get_user_input():
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Client Age', min_value=0, max_value=100, value=30, step=1)
            gender = st.selectbox('Client Gender Identity', 
                                options=[k for k in GENDER_MAPPING.keys() if k not in ["Femalefemale", "Malemale"]])
            ethnicity = st.selectbox('Client Ethnicity', 
                                   options=[k for k in ETHNICITY_MAPPING.keys() if not any(x in k for x in [",", "_"])])
            disability = st.selectbox('Client Disability', 
                                    options=["Unknown", "no", "yes", "prefer_not_to_answer", "do_not_know", "did_not_ask"])
            agency = st.selectbox('Visited Agency',
                                options=list(AGENCIES.keys()),
                                help="Start typing to search for an agency")
        
        with col2:
            self_identifies = st.selectbox('Client Self-Identifies As',
                                         options=[k for k in SELF_IDENTIFIES_MAPPING.keys()])
            household_size = st.number_input('Household Size', 
                                           min_value=1, value=2, step=1)
            income = st.number_input('Monthly Household Income', 
                                   min_value=0.0, value=2000.0, step=100.0)
            prev_visits = st.number_input('Number of Visits Last Month', 
                                        min_value=0.0, max_value=31.0, value=0.0, step=0.1,
                                        help="Enter the average number of visits per month for clustering")
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = {
                'Client Age': age,
                'Client Gender Identity-Labels': GENDER_MAPPING[gender],
                'Client Ethnicity-Labels': ETHNICITY_MAPPING[ethnicity],
                'Client Disability': DISABILITY_MAPPING[disability],
                'Client Self-Identifies As': SELF_IDENTIFIES_MAPPING[self_identifies],
                'Household Size': household_size,
                'Monthly Household Income': income,
                'Visited Agency': AGENCIES[agency],
                'Avg Visits Per Month': float(prev_visits)  # Include this directly in input_data
            }
            return submitted, input_data
        
        return False, None

def main():
    st.title('Food Bank Visit Prediction')
    
    # Clear any stored predictions if the page is refreshed
    if 'cluster_prediction' in st.session_state:
        del st.session_state['cluster_prediction']
    if 'visit_prediction' in st.session_state:
        del st.session_state['visit_prediction']
    
    # Load models
    kmeans_model, decision_tree_model = load_models()
    if not (kmeans_model and decision_tree_model):
        return
    
    # Get user input
    submitted, input_data = get_user_input()
    
    if submitted:
        try:
            # Show raw input
            #st.write("Raw input data:")
            #st.write(input_data)
            
            # Preprocess input data
            clustering_data, decision_tree_data = preprocess_data(input_data)
            
            # Display processed input for verification
            with st.expander("View Processed Input Data"):
                st.write("Clustering Data (includes Avg Visits Per Month):")
                st.write(clustering_data)
                st.write(f"Clustering shape: {clustering_data.shape}")
                st.write("\nDecision Tree Data:")
                st.write(decision_tree_data)
                st.write(f"Decision Tree shape: {decision_tree_data.shape}")
            
           # Make predictions
            with st.spinner('Making predictions...'):
                # Get predictions and distances
                distances = kmeans_model.transform(clustering_data)
                cluster_prediction = kmeans_model.predict(clustering_data)
                visit_prediction = decision_tree_model.predict(decision_tree_data)
                
                # Display results with larger headers and full width
                st.success("Prediction Complete!")
                
                # Get cluster info
                cluster_info = CLUSTER_DESCRIPTIONS[cluster_prediction[0]]
                
                # Header row with larger text and better spacing
                st.markdown(f"""
                <div style='padding: 1em 0; text-align: center;'>
                    <h2>Predicted Cluster {cluster_prediction[0]}</h2>
                    <h3>{cluster_info['title']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics in columns with larger headers
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Cluster Assignment")
                    st.write(f"This client matches characteristics of Cluster {cluster_prediction[0]}")
                with col2:
                    st.markdown("### Visit Prediction")
                    st.markdown(f"#### {visit_prediction[0]:.1f} visits per month")
                
                # Display cluster characteristics
                st.write("### Cluster Profile")
                profile_col1, profile_col2 = st.columns(2)
                
                with profile_col1:
                    st.write("**Cluster Characteristics:**")
                    characteristics = pd.DataFrame({
                        'Metric': ['Average Age', 'Predominant Gender', 'Typical Ethnicity', 'Average Household Size'],
                        'Value': [
                            f"{cluster_info['avg_age']:.1f}",
                            cluster_info['gender'],
                            cluster_info['ethnicity'],
                            f"{cluster_info['household_size']:.1f}"
                        ]
                    })
                    st.dataframe(characteristics, hide_index=True, width=None)  # width=None allows full width
                
                with profile_col2:
                    st.write("**Key Metrics:**")
                    metrics = pd.DataFrame({
                        'Metric': ['Monthly Income', 'Typical Visits/Month', 'Disability Rate'],
                        'Value': [
                            cluster_info['income'],
                            f"{cluster_info['visits']:.2f}",
                            f"{cluster_info['disability']:.2f}"
                        ]
                    })
                    st.dataframe(metrics, hide_index=True, width=None)  # width=None allows full width
                
                st.markdown(f"""
                <div style='padding: 1em; background-color: #f0f2f6; border-radius: 5px;'>
                    <strong>Key Observations:</strong> {cluster_info['key_obs']}
                </div>
                """, unsafe_allow_html=True)
                
                             # Continue with your existing feature influence analysis...
                st.write("### Detailed Analysis")
                
                # Create analysis DataFrames
                comparison_df = pd.DataFrame({
                    'Feature': clustering_data.columns,
                    'Current Value': clustering_data.iloc[0].values,
                    'Cluster Center': kmeans_model.cluster_centers_[cluster_prediction[0]],
                    'Difference': abs(clustering_data.iloc[0].values - kmeans_model.cluster_centers_[cluster_prediction[0]])
                })
                comparison_df = comparison_df.sort_values('Difference', ascending=False)
                
                distances_df = pd.DataFrame({
                    'Cluster': range(len(distances[0])),
                    'Distance': distances[0],
                    'Relative Distance': distances[0] / np.min(distances[0])
                })
                
                # Display analysis side by side
                st.write("### Cluster Analysis")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.write("**Most Influential Features:**")
                    st.dataframe(comparison_df[['Feature', 'Difference']].head(),
                               hide_index=True)
                
                with col4:
                    st.write("**Distances to Clusters:**")
                    st.dataframe(distances_df.style.highlight_min('Distance'),
                               hide_index=True)
                
                # Show detailed comparison in expandable section
                with st.expander("View Detailed Feature Comparison"):
                    st.dataframe(comparison_df[['Feature', 'Current Value', 'Cluster Center', 'Difference']],
                               hide_index=True)
            
                        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Debug info:", e.__class__.__name__)
            if st.checkbox("Show detailed error information"):
                st.exception(e)

if __name__ == "__main__":
    main()