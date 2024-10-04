import pandas as pd

# Function to load dataset
def load_data(file_path):
    # Load data into pandas DataFrame
    data = pd.read_csv(file_path)
    return data

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

# Function to export dataset to a new file
def export_data(df, output_file_path):
    df.to_csv(output_file_path, index=False)
    print(f"Data exported to {output_file_path}")

# Example usage:
if __name__ == "__main__":
    # Specify the file path to load data from
    file_path = r'D:\Study Material\annual-enterprise-survey-2023-financial-year-provisional.csv'  
    df = load_data(file_path)
    
    # Show dataset details
    show_data_details(df)
    
    # Export the dataset to a new file
    export_file_path = 'exported_data.csv'
    export_data(df, export_file_path)
