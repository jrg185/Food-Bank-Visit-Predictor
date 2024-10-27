import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
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

GROUP_INFO = {
    "Very High": {
        "size": "16.3%",
        "visits": {"avg": 5.0, "range": "3.4-13.0"},
        "demographics": {
            "female": "73.6%",
            "white": "32.5%",
            "unknown": "24.9%",
            "black": "20.9%",
            "disability": "40.1%",
            "disability_veteran": "43.9%"
        },
        "household": "3.8 members (22% above avg)",
        "income": "$1,524 (20% above avg)",
        "key_characteristics": [
            "Highest visit frequency (5 visits/month)",
            "Most diverse age distribution",
            "Highest female percentage (73.6%)",
            "Highest disability rate (40.1%)",
            "Highest disability/veteran identification (43.9%)",
            "Larger households: 3.8 members",
            "Higher income: $1,524/month"
        ]
    },
    "High": {
        "size": "16.9%",
        "visits": {"avg": 2.9, "range": "2.4-3.4"},
        "demographics": {
            "female": "70.5%",
            "white": "34.4%",
            "unknown": "25.3%",
            "black": "17.8%",
            "disability": "29.8%",
            "disability_veteran": "36.2%"
        },
        "household": "3.5 members (13% above avg)",
        "income": "$1,397 (10% above avg)",
        "key_characteristics": [
            "High visit frequency (2.9 visits/month)",
            "Highest White/Anglo percentage (34.4%)",
            "Above average disability/veteran identification (36.2%)",
            "Larger households: 3.5 members",
            "Higher income: $1,397/month"
        ]
    },
    "Medium-High": {
        "size": "16.8%",
        "visits": {"avg": 2.2, "range": "1.9-2.4"},
        "demographics": {
            "female": "71.9%",
            "white": "31.6%",
            "unknown": "27.7%",
            "black": "15.5%",
            "disability": "30.3%",
            "disability_veteran": "35.1%"
        },
        "household": "3.2 members (3% above avg)",
        "income": "$1,334 (5% above avg)",
        "key_characteristics": [
            "Moderate-high visits (2.2 visits/month)",
            "High female percentage (71.9%)",
            "Average household size: 3.2 members",
            "Slightly higher income: $1,334/month"
        ]
    },
    "Medium-Low": {
        "size": "16.5%",
        "visits": {"avg": 1.7, "range": "1.6-1.9"},
        "demographics": {
            "female": "70.5%",
            "white": "31.4%",
            "unknown": "25.4%",
            "black": "15.7%",
            "disability": "30.1%",
            "disability_veteran": "30.4%"
        },
        "household": "3.0 members (avg)",
        "income": "$1,207 (5% below avg)",
        "key_characteristics": [
            "Moderate-low visits (1.7 visits/month)",
            "Average household size: 3.0 members",
            "Slightly lower income: $1,207/month"
        ]
    },
    "Low": {
        "size": "16.9%",
        "visits": {"avg": 1.4, "range": "1.3-1.6"},
        "demographics": {
            "female": "68.3%",
            "white": "30.6%",
            "unknown": "23.5%",
            "black": "15.8%",
            "disability": "30.4%",
            "disability_veteran": "30.6%"
        },
        "household": "2.8 members (10% below avg)",
        "income": "$1,143 (10% below avg)",
        "key_characteristics": [
            "Low visits (1.4 visits/month)",
            "Smaller households: 2.8 members",
            "Lower income: $1,143/month"
        ]
    },
    "Very Low": {
        "size": "16.7%",
        "visits": {"avg": 1.1, "range": "1.0-1.3"},
        "demographics": {
            "female": "67.9%",
            "white": "25.4%",
            "unknown": "26.8%",
            "black": "18.4%",
            "disability": "28.2%",
            "disability_veteran": "26.9%"
        },
        "household": "2.5 members (19% below avg)",
        "income": "$1,016 (20% below avg)",
        "key_characteristics": [
            "Lowest visits (1.1 visits/month)",
            "Different ethnicity pattern",
            "Highest \"no disability\" rate",
            "Smallest households: 2.5 members",
            "Lowest income: $1,016/month"
        ]
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
        # Load KNN dictionary and extract components
        knn_dict = joblib.load("knn_model.pkl")
        knn_model = knn_dict["model"]
        scaler = knn_dict["scaler"]
        
        # Load decision tree
        decision_tree_model = joblib.load("the_best_tree_model.pkl")
        
        st.success("Models loaded successfully!")
        return knn_model, decision_tree_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def preprocess_data(input_dict, scaler):
    """Preprocess the input data for model prediction"""
    # Create base DataFrame from input
    df = pd.DataFrame([input_dict])
    
    # Define exact feature order and dtypes from your training data
    feature_dtypes = {
        'Client Age': 'float64',
        'Client Gender Identity-Labels': 'int32',
        'Client Ethnicity-Labels': 'int32',
        'Client Disability': 'int32',
        'Client Self-Identifies As': 'int32',
        'Household Size': 'int64',
        'Monthly Household Income': 'float64',
        'Visited Agency': 'int32'
    }
    
    # Convert each column to the correct dtype
    for col, dtype in feature_dtypes.items():
        df[col] = df[col].astype(dtype)
    
    # Ensure columns are in correct order
    input_data = df[feature_dtypes.keys()].copy()
    
    # Scale the data for KNN
    knn_data_scaled = pd.DataFrame(
        scaler.transform(input_data),
        columns=input_data.columns
    )
    
    # Use the same data (unscaled) for decision tree
    decision_tree_data = input_data
    
    return knn_data_scaled, decision_tree_data

def get_user_input():
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Client Age', min_value=0.0, max_value=100.0, value=30.0, step=1.0)
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
                                        help="Enter the average number of visits per month")
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = {
                'Client Age': float(age),  # Ensure float64
                'Client Gender Identity-Labels': int(GENDER_MAPPING[gender]),  # Ensure int32
                'Client Ethnicity-Labels': int(ETHNICITY_MAPPING[ethnicity]),  # Ensure int32
                'Client Disability': int(DISABILITY_MAPPING[disability]),  # Ensure int32
                'Client Self-Identifies As': int(SELF_IDENTIFIES_MAPPING[self_identifies]),  # Ensure int32
                'Household Size': int(household_size),  # Ensure int64
                'Monthly Household Income': float(income),  # Ensure float64
                'Visited Agency': int(AGENCIES[agency])  # Ensure int32
            }
            return submitted, input_data
        
        return False, None

def determine_group(visits):
    """Determine which group a client belongs to based on predicted visits"""
    if visits >= 3.4:
        return "Very High"
    elif visits >= 2.4:
        return "High"
    elif visits >= 1.9:
        return "Medium-High"
    elif visits >= 1.6:
        return "Medium-Low"
    elif visits >= 1.3:
        return "Low"
    else:
        return "Very Low"

def display_group_info(visits):
    """Display detailed group information based on predicted visits"""
    group_name = determine_group(visits)
    group_data = GROUP_INFO[group_name]
    
    st.markdown(f"""
    ### Group Profile: {group_name}
    
    **Group Size:** {group_data['size']} of total clients  
    **Visit Pattern:** Average {group_data['visits']['avg']} visits/month (range: {group_data['visits']['range']})
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Demographics")
        demographics_df = pd.DataFrame({
            'Metric': [
                'Female',
                'White/Anglo',
                'Unknown Ethnicity',
                'Black/African American',
                'Disability',
                'Disability/Veteran ID'
            ],
            'Percentage': [
                group_data['demographics']['female'],
                group_data['demographics']['white'],
                group_data['demographics']['unknown'],
                group_data['demographics']['black'],
                group_data['demographics']['disability'],
                group_data['demographics']['disability_veteran']
            ]
        })
        st.dataframe(demographics_df, hide_index=True)
    
    with col2:
        st.markdown("#### Household Characteristics")
        household_df = pd.DataFrame({
            'Characteristic': ['Household Size', 'Monthly Income'],
            'Level': [group_data['household'], group_data['income']]
        })
        st.dataframe(household_df, hide_index=True)
    
    st.markdown("#### Key Characteristics")
    for char in group_data['key_characteristics']:
        st.markdown(f"• {char}")

def display_predictions_and_analysis(knn_predicted_visits, dt_predicted_visits, group_name, group_data):
    """Display all predictions and analysis in a clean, centered format"""
    
    # Success banner
    st.markdown("""
        <div style='background-color: #28a745; color: white; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 20px;'>
            <h4 style='margin: 0;'>Prediction Complete!</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Predictions section
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <div style='display: inline-block; width: 45%; margin: 0 2%;'>
                <h4>KNN Visit Prediction</h4>
                <h2>{:.1f} visits per month</h2>
                <p style='color: #666;'>(Based on similar clients)</p>
            </div>
            <div style='display: inline-block; width: 45%; margin: 0 2%;'>
                <h4>Decision Tree Prediction</h4>
                <h2>{:.1f} visits per month</h2>
                <p style='color: #666;'>(Based on client characteristics)</p>
            </div>
        </div>
    """.format(knn_predicted_visits, dt_predicted_visits), unsafe_allow_html=True)
    
    # Group Profile Header
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h3 style='color: #333; margin-bottom: 20px;'>Similar Clients Analysis</h3>
            <h4 style='color: #444; margin-bottom: 15px;'>Group Profile: {}</h4>
            <div style='margin-bottom: 10px;'>
                <strong>Group Size:</strong> {} of total clients
            </div>
            <div style='margin-bottom: 20px;'>
                <strong>Visit Pattern:</strong> Average {} visits/month (range: {})
            </div>
        </div>
    """.format(
        group_name,
        group_data['size'],
        group_data['visits']['avg'],
        group_data['visits']['range']
    ), unsafe_allow_html=True)
    
    # Demographics and Household Characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='text-align: center; color: #444;'>Demographics</h4>", unsafe_allow_html=True)
        demographics_df = pd.DataFrame({
            'Metric': [
                'Female',
                'White/Anglo',
                'Unknown Ethnicity',
                'Black/African American',
                'Disability',
                'Disability/Veteran ID'
            ],
            'Percentage': [
                group_data['demographics']['female'],
                group_data['demographics']['white'],
                group_data['demographics']['unknown'],
                group_data['demographics']['black'],
                group_data['demographics']['disability'],
                group_data['demographics']['disability_veteran']
            ]
        })
        st.dataframe(demographics_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='text-align: center; color: #444;'>Household Characteristics</h4>", unsafe_allow_html=True)
        household_df = pd.DataFrame({
            'Characteristic': ['Household Size', 'Monthly Income'],
            'Level': [group_data['household'], group_data['income']]
        })
        st.dataframe(household_df, hide_index=True, use_container_width=True)
    
    # Key Characteristics
    st.markdown("<h4 style='text-align: center; color: #444; margin-top: 20px;'>Key Characteristics</h4>", unsafe_allow_html=True)
    char_html = "<div style='text-align: center; margin-bottom: 20px;'>"
    for char in group_data['key_characteristics']:
        char_html += f"<p style='margin: 5px 0;'>• {char}</p>"
    char_html += "</div>"
    st.markdown(char_html, unsafe_allow_html=True)

def main():
    st.title('Food Bank Visit Prediction')
    
    # Load models and scaler
    knn_model, decision_tree_model, scaler = load_models()
    if not (knn_model and decision_tree_model and scaler):
        return
    
    # Get user input
    submitted, input_data = get_user_input()
    
    if submitted:
        try:
            # Preprocess input data
            knn_data, decision_tree_data = preprocess_data(input_data, scaler)
            
            # Optional data verification expander
            with st.expander("View Processed Input Data"):
                st.write("KNN Input Data (Scaled):")
                st.write(knn_data)
                st.write("\nDecision Tree Data:")
                st.write(decision_tree_data)
            
            # Make predictions
            with st.spinner('Making predictions...'):
                # Get predictions
                knn_predicted_visits = knn_model.predict(knn_data)[0]
                dt_predicted_visits = decision_tree_model.predict(decision_tree_data)[0]
                
                # Get nearest neighbors info for distance analysis
                distances, indices = knn_model.kneighbors(knn_data)
                
                # Get group information
                group_name = determine_group(knn_predicted_visits)
                group_data = GROUP_INFO[group_name]
                
                # Display main predictions and analysis
                display_predictions_and_analysis(
                    knn_predicted_visits,
                    dt_predicted_visits,
                    group_name,
                    group_data
                )
                
                # Additional Analysis Section
                st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
                st.markdown("""
                    <div style='text-align: center; margin-bottom: 20px;'>
                        <h3>Detailed Analysis</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Distance Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h4 style='text-align: center;'>Distance Metrics</h4>", 
                              unsafe_allow_html=True)
                    neighbors_df = pd.DataFrame({
                        'Neighbor': range(1, len(distances[0]) + 1),
                        'Distance': [f"{d:.3f}" for d in distances[0]]
                    })
                    st.dataframe(neighbors_df.head(),
                               hide_index=True,
                               use_container_width=True)
                
                with col2:
                    st.markdown("<h4 style='text-align: center;'>Statistical Summary</h4>", 
                              unsafe_allow_html=True)
                    similarity_stats = pd.DataFrame({
                        'Metric': ['Closest Match', 'Average Distance', 'Furthest Match'],
                        'Value': [
                            f"{np.min(distances[0]):.3f}",
                            f"{np.mean(distances[0]):.3f}",
                            f"{np.max(distances[0]):.3f}"
                        ]
                    })
                    st.dataframe(similarity_stats,
                               hide_index=True,
                               use_container_width=True)
                
                # Feature Analysis
                with st.expander("View Feature Analysis"):
                    st.markdown("<h4 style='text-align: center;'>Feature Details</h4>", 
                              unsafe_allow_html=True)
                    
                    # Create feature comparison
                    feature_diff_df = pd.DataFrame({
                        'Feature': knn_data.columns,
                        'Scaled Value': [f"{v:.3f}" for v in knn_data.iloc[0].values],
                        'Original Value': decision_tree_data.iloc[0].values
                    }).sort_values('Feature')
                    
                    st.dataframe(feature_diff_df,
                               hide_index=True,
                               use_container_width=True)
                
                # Optional: Add explanatory note at the bottom
                st.markdown("""
                    <div style='text-align: center; margin-top: 30px; padding: 10px; 
                              background-color: #f8f9fa; border-radius: 5px;'>
                        <p style='color: #666; margin: 0;'>
                            Predictions are based on historical patterns and similar client profiles. 
                            Actual visit patterns may vary.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Debug info:", e.__class__.__name__)
            if st.checkbox("Show detailed error information"):
                st.exception(e)

if __name__ == "__main__":
    main()
