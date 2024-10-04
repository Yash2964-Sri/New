import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load dataset
def load_data(file_path):
    # Loading data into pandas DataFrame
    data = pd.read_csv(file_path)
    return data

# Function to perform Exploratory data analysis(EDA)
def perform_eda(df):
    print("### BASIC DATA INFORMATION ###")
    print("\n1. First few rows of the dataset:")
    print(df.head(), "\n")

    print("\n2. Basic information about the dataset:")
    df.info()  
    print("\n")

    print("\n3. Summary statistics for numerical columns:")
    print(df.describe(), "\n")

    print("\n4. Summary statistics for categorical columns:")
    print(df.describe(include='object'), "\n")

    print("\n5. Checking for missing values:")
    print(df.isnull().sum(), "\n")

    # Checking dataset shape
    print(f"\n6. Dataset shape: {df.shape} (Rows, Columns)\n")
    
    # Checking unique values for categorical columns
    print("7. Unique values in categorical columns:")
    for col in df.select_dtypes(include='object').columns:
        print(f"\nUnique values in {col}:")
        print(df[col].value_counts(), "\n")
    
    # Pairwise correlation for numerical columns
    print("\n8. Correlation between numerical features:")
    correlation_matrix = df.corr(numeric_only=True)
    print(correlation_matrix, "\n")
    
    return correlation_matrix

# Function to visualize important insights
def visualize_data(df):
    # Plotting correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

    # Plotting missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values in Dataset")
    plt.show()

    # Plotting distribution of numerical columns
    df.select_dtypes(include='number').hist(bins=15, figsize=(15, 10))
    plt.suptitle("Distribution of Numerical Features", size=18)
    plt.show()

    # Plotting pairplot for numerical columns
    if df.select_dtypes(include='number').shape[1] > 1:
        sns.pairplot(df.select_dtypes(include='number'))
        plt.suptitle("Pairplot of Numerical Features", size=18)
        plt.show()

# Function to export dataset to a new file
def export_data(df, output_file_path):
    df.to_csv(output_file_path, index=False)
    print(f"Data exported to {output_file_path}")

# Example usage:
if __name__ == "__main__":
    # Specify the file path to load data from
    file_path = r'D:\Study Material\annual-enterprise-survey-2023-financial-year-provisional.csv'
    
    # Load data
    df = load_data(file_path)
    
    # Perform EDA
    correlation_matrix = perform_eda(df)
    
    # Visualize the data
    visualize_data(df)
    
    # Export the dataset to a new file (optional)
    export_file_path = 'exported_data.csv'
    export_data(df, export_file_path)
