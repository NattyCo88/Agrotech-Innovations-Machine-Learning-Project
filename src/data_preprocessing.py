import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations

def preprocess_data(input_path, output_path):
    # Load data from the CSV file
    agri_data = pd.read_csv(input_path)
    
    # Convert Nutrient columns to numeric
    for col in ['Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)']:
        agri_data[col] = pd.to_numeric(agri_data[col], errors='coerce')

    # Drop columns with high % of missing values
    agri_data_cleaned = agri_data.drop(columns=['Humidity Sensor (%)'])

    # Impute missing values with the mean
    columns_to_impute = ['Temperature Sensor (°C)', 'Water Level Sensor (mm)', 
                         'Nutrient N Sensor (ppm)', 'Light Intensity Sensor (lux)', 
                         'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)']
    for col in columns_to_impute:
        agri_data_cleaned[col] = agri_data_cleaned[col].fillna(agri_data_cleaned[col].mean())

    # Apply capitalization to all object-type columns
    for column in agri_data_cleaned.select_dtypes(include=['object']).columns:
        agri_data_cleaned[column] = agri_data_cleaned[column].str.title()

    # Filter out rows with negative values and create a new copy
    numeric_columns = agri_data_cleaned.select_dtypes(include=['number']).columns
    agri_data_cleaned_filtered = agri_data_cleaned.loc[(agri_data_cleaned[numeric_columns] >= 0).all(axis=1)].copy()

    # Encoding categorical columns with mappings
    # Plant Stage Encoding
    stage_mapping = {'Seedling': 0, 'Vegetative': 1, 'Maturity': 2}
    agri_data_cleaned_filtered['Plant Stage Encoded'] = agri_data_cleaned_filtered['Plant Stage'].map(stage_mapping)

    # Plant Type Encoding
    type_mapping = {'Vine Crops': 0, 'Herbs': 1, 'Fruiting Vegetables': 2, 'Leafy Greens': 3}
    agri_data_cleaned_filtered['Plant Type Encoded'] = agri_data_cleaned_filtered['Plant Type'].map(type_mapping)
    agri_data_cleaned_filtered['Prev Plant Type Encoded'] = agri_data_cleaned_filtered['Previous Cycle Plant Type'].map(type_mapping)

    # System Location Encoding
    zone_mapping = {'Zone_D': 0, 'Zone_G': 1, 'Zone_F': 2, 'Zone_B': 3, 'Zone_C': 4, 'Zone_A': 5, 'Zone_E': 6}
    agri_data_cleaned_filtered['System Location Code Encoded'] = agri_data_cleaned_filtered['System Location Code'].map(zone_mapping)

    # Feature Engineering
    # Light and CO2 Combination
    agri_data_cleaned_filtered['Light_per_CO2'] = (
        agri_data_cleaned_filtered['Light Intensity Sensor (lux)'] / agri_data_cleaned_filtered['CO2 Sensor (ppm)']
    )
    # Electrical Conductivity and Nutrients
    agri_data_cleaned_filtered['EC_to_N'] = (
        agri_data_cleaned_filtered['EC Sensor (dS/m)'] / agri_data_cleaned_filtered['Nutrient N Sensor (ppm)']
    )
    agri_data_cleaned_filtered['EC_to_P'] = (
        agri_data_cleaned_filtered['EC Sensor (dS/m)'] / agri_data_cleaned_filtered['Nutrient P Sensor (ppm)']
    )
    agri_data_cleaned_filtered['EC_to_K'] = (
        agri_data_cleaned_filtered['EC Sensor (dS/m)'] / agri_data_cleaned_filtered['Nutrient K Sensor (ppm)']
    )

    # Water and Nutrients
    agri_data_cleaned_filtered['Total_Nutrients'] = (
        agri_data_cleaned_filtered['Nutrient N Sensor (ppm)'] +
        agri_data_cleaned_filtered['Nutrient P Sensor (ppm)'] +
        agri_data_cleaned_filtered['Nutrient K Sensor (ppm)']
    )
    agri_data_cleaned_filtered['Water_to_Nutrients'] = (
        agri_data_cleaned_filtered['Water Level Sensor (mm)'] / agri_data_cleaned_filtered['Total_Nutrients']
    )

    # Normalized Features
    agri_data_cleaned_filtered['Nutrient_N_Normalized'] = (
        agri_data_cleaned_filtered['Nutrient N Sensor (ppm)'] / agri_data_cleaned_filtered['Total_Nutrients']
    )
    agri_data_cleaned_filtered['Nutrient_P_Normalized'] = (
        agri_data_cleaned_filtered['Nutrient P Sensor (ppm)'] / agri_data_cleaned_filtered['Total_Nutrients']
    )
    agri_data_cleaned_filtered['Nutrient_K_Normalized'] = (
        agri_data_cleaned_filtered['Nutrient K Sensor (ppm)'] / agri_data_cleaned_filtered['Total_Nutrients']
    )
    agri_data_cleaned_filtered['Log_Light_Intensity'] = np.log1p(agri_data_cleaned_filtered['Light Intensity Sensor (lux)'])
    agri_data_cleaned_filtered['Log_CO2'] = np.log1p(agri_data_cleaned_filtered['CO2 Sensor (ppm)'])

    # Apply log function on light to CO2 readings:
    agri_data_cleaned_filtered['Log_Light_per_CO2'] = agri_data_cleaned_filtered['Log_Light_Intensity'] / agri_data_cleaned_filtered['Log_CO2']
    
        # Drop object type columns
    object_columns = agri_data_cleaned_filtered.select_dtypes(include=['object']).columns
    agri_data_cleaned_filtered = agri_data_cleaned_filtered.drop(columns=object_columns)

    # Save preprocessed data
    agri_data_cleaned_filtered.to_csv(output_path, index=False)

    print(agri_data_cleaned_filtered.head(10))
    print(agri_data_cleaned_filtered.describe())
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_path = "Data/agri_data.csv"  # Path to the input file
    output_path = "Data/agri_data_cleaned.csv"  # Path to save the cleaned data
    
    # Preprocess data
    preprocess_data(input_path, output_path)
