import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set style for better visualizations
sns.set_theme()  # Using seaborn's default theme

def load_data():
    """
    Load the Titanic dataset
    Returns:
        pandas DataFrame: The loaded dataset
    """
    try:
        # Update the path to where you downloaded the Titanic dataset
        df = pd.read_csv('data/train.csv')
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print("Error: Please ensure the Titanic dataset is in the data directory")
        return None

def display_basic_info(df):
    """
    Display basic information about the dataset
    Args:
        df: pandas DataFrame
    """
    print("\nBasic Dataset Information:")
    print("-" * 30)
    print("\nShape of the dataset:", df.shape)
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe())

def clean_data(df):
    """
    Clean the dataset by handling missing values and converting datatypes
    Args:
        df: pandas DataFrame
    Returns:
        pandas DataFrame: Cleaned dataset
    """
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()
    
    # Handle missing values
    # Fill missing age values with median
    age_imputer = SimpleImputer(strategy='median')
    df_clean['Age'] = age_imputer.fit_transform(df_clean[['Age']])
    
    # Fill missing embarked values with mode
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    
    # Fill missing cabin values with 'Unknown'
    df_clean['Cabin'].fillna('Unknown', inplace=True)
    
    # Convert categorical variables
    df_clean['Sex'] = pd.Categorical(df_clean['Sex']).codes
    df_clean['Embarked'] = pd.Categorical(df_clean['Embarked']).codes
    
    return df_clean

def analyze_survival_patterns(df):
    """
    Analyze survival patterns across different features
    Args:
        df: pandas DataFrame
    """
    # Survival by Sex
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title('Survival Rate by Gender')
    plt.show()
    
    # Survival by Passenger Class
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title('Survival Rate by Passenger Class')
    plt.show()
    
    # Age distribution of survivors vs non-survivors
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Survived', y='Age', data=df)
    plt.title('Age Distribution by Survival Status')
    plt.show()
    
    # Survival by Age Groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], 
                           labels=['Child', 'Teen', 'Adult', 'Middle Aged', 'Senior'])
    plt.figure(figsize=(12, 6))
    sns.barplot(x='AgeGroup', y='Survived', data=df)
    plt.title('Survival Rate by Age Group')
    plt.show()

def analyze_correlations(df):
    """
    Analyze correlations between numerical variables
    Args:
        df: pandas DataFrame
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.show()

def main():
    # Load the data
    df = load_data()
    if df is None:
        return
    
    # Display basic information
    display_basic_info(df)
    
    # Clean the data
    df_clean = clean_data(df)
    
    # Analyze survival patterns
    analyze_survival_patterns(df)
    
    # Analyze correlations
    analyze_correlations(df_clean)

if __name__ == "__main__":
    main()
