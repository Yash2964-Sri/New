import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load dataset
def load_data(file_path):
    # Load data into pandas DataFrame
    data = pd.read_csv(file_path)
    return data

# Function to handle missing values
def handle_missing_values(df):
    print("\n### Handling Missing Values ###")

    # Show missing values count
    missing_values = df.isnull().sum()
    print("Missing Values Before Handling:")
    print(missing_values)

    # Handling missing values: Example using mean/median/mode for numerical/categorical columns
    for col in df.columns:
        if df[col].dtype == 'object':  # If categorical
            df[col].fillna(df[col].mode()[0], inplace=True)  # Fill with mode
        else:  # If numerical
            df[col].fillna(df[col].median(), inplace=True)  # Fill with median

    print("\nMissing Values After Handling:")
    print(df.isnull().sum())

# Function to detect and handle outliers using IQR
def handle_outliers(df):
    print("\n### Handling Outliers ###")
    
    # Detect outliers for numerical columns using IQR method
    numerical_columns = df.select_dtypes(include='number')
    
    for col in numerical_columns.columns:
        Q1 = df[col].quantile(0.25)  # 25th percentile
        Q3 = df[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1  # Interquartile Range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identifying outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\n{col}: Found {len(outliers)} outliers")

        # Option 1: Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print("\nDataset after handling outliers:")
    print(df.describe())  # Show summary statistics after outlier handling

    return df

# Function to display dataset details
def show_data_details(df):
    # Display basic information about the dataset
    print("Number of Rows: ", df.shape[0])
    print("Number of Columns: ", df.shape[1])
    
    # Display first five rows
    print("\nFirst 5 Rows:")
    print(df.head())
    
    # Display dataset size (number of elements)
    print("\nDataset Size: ", df.size)
    
    # Display number of missing values
    print("\nMissing Values in Each Column:")
    print(df.isnull().sum())
    
    # Display summary statistics for numerical columns
    numerical_columns = df.select_dtypes(include='number')
    
    if not numerical_columns.empty:
        print("\nSum of Numerical Columns:")
        print(numerical_columns.sum())
        
        print("\nAverage of Numerical Columns:")
        print(numerical_columns.mean())
        
        print("\nMin Values in Numerical Columns:")
        print(numerical_columns.min())
        
        print("\nMax Values in Numerical Columns:")
        print(numerical_columns.max())
    else:
        print("\nNo numerical columns found in the dataset.")

# Function to visualize data before and after handling outliers
def visualize_data(df, column):
    if column in df.columns and df[column].dtype in ['int64', 'float64']:
        # Boxplot for outlier detection
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[column])
        plt.title(f"Boxplot of {column}")
        plt.show()
    else:
        print(f"Column '{column}' is either not in the dataset or is not numerical.")

# Function to export dataset to a new file
def export_data(df, output_file_path):
    df.to_csv(output_file_path, index=False)
    print(f"Data exported to {output_file_path}")

# Example usage:
if __name__ == "__main__":
    # Specify the file path to load data from
    file_path = r'D:\Study Material\annual-enterprise-survey-2023-financial-year-provisional.csv'  # Use raw string to handle backslashes
    
    # Load data
    df = load_data(file_path)
    
    # Show dataset details before processing
    show_data_details(df)
    
    # Handle missing values
    handle_missing_values(df)

    # Print available columns for debugging
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())

    # Visualize data before handling outliers (optional)
    column_to_visualize = 'Value'  # Replace 'Value' with the actual column name you want to visualize
    visualize_data(df, column_to_visualize)  
    
    # Handle outliers
    df_cleaned = handle_outliers(df)

    # Visualize data after handling outliers (optional)
    visualize_data(df_cleaned, column_to_visualize)  
    
    # Export the cleaned dataset to a new file
    export_file_path = 'cleaned_data.csv'
    export_data(df_cleaned, export_file_path)
