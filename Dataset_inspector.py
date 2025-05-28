# dataset_inspector.py
import pandas as pd
import numpy as np

def inspect_dataset(file_path='Student Mental Health.csv'):
    """Inspect the dataset to understand its structure"""
    try:
        print("🔍 Dataset Inspector")
        print("=" * 50)
        
        # Load dataset
        df = pd.read_csv(file_path)
        
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📋 Number of rows: {df.shape[0]}")
        print(f"📋 Number of columns: {df.shape[1]}")
        print()
        
        print("📝 Column Names:")
        print("-" * 30)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. '{col}'")
        print()
        
        print("🔢 Data Types:")
        print("-" * 30)
        print(df.dtypes)
        print()
        
        print("📈 Basic Statistics:")
        print("-" * 30)
        print(df.describe(include='all'))
        print()
        
        print("❓ Missing Values:")
        print("-" * 30)
        missing = df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")
        print()
        
        print("🎯 Unique Values per Column:")
        print("-" * 30)
        for col in df.columns:
            unique_count = df[col].nunique()
            print(f"{col}: {unique_count} unique values")
            if unique_count <= 10:  # Show unique values if less than 10
                print(f"   Values: {list(df[col].unique())}")
        print()
        
        # Try to identify target column
        print("🎯 Potential Target Columns:")
        print("-" * 30)
        depression_cols = [col for col in df.columns if 'depression' in col.lower()]
        mental_health_cols = [col for col in df.columns if any(word in col.lower() 
                             for word in ['mental', 'health', 'anxiety', 'panic', 'treatment'])]
        
        if depression_cols:
            print("Depression-related columns:")
            for col in depression_cols:
                print(f"  - {col}")
                print(f"    Values: {list(df[col].unique())}")
        
        if mental_health_cols:
            print("Mental health-related columns:")
            for col in mental_health_cols:
                print(f"  - {col}")
                print(f"    Values: {list(df[col].unique())}")
        
        print()
        print("✅ Dataset inspection completed!")
        print("💡 Use this information to update your Streamlit form fields.")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        print("Please make sure you have downloaded the dataset from Kaggle")
        print("and placed it in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error inspecting dataset: {str(e)}")
        return None

if __name__ == "__main__":
    df = inspect_dataset()
    
    if df is not None:
        print("\n" + "="*50)
        print("📋 SAMPLE DATA (First 5 rows):")
        print("="*50)
        print(df.head())
        
        print("\n" + "="*50)
        print("🔧 SUGGESTED STREAMLIT FORM FIELDS:")
        print("="*50)
        
        # Generate suggested form fields based on actual columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        print("\n# Categorical Fields:")
        for col in categorical_cols:
            unique_vals = list(df[col].unique())[:10]  # First 10 unique values
            print(f"# {col}: {unique_vals}")
        
        print("\n# Numerical Fields:")
        for col in numerical_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"# {col}: Range {min_val} to {max_val}")