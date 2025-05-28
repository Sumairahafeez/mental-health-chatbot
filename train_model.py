# student_mental_health_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class StudentMentalHealthPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        self.best_model = None
        self.best_model_name = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the student mental health dataset"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Handle missing values
        df = df.dropna()
        
        # Identify the target column - likely to be "Do you have Depression?"
        possible_target_cols = [col for col in df.columns if 'depression' in col.lower()]
        
        if possible_target_cols:
            target_col = possible_target_cols[0]
        else:
            # If no depression column, look for other mental health indicators
            possible_targets = [col for col in df.columns if any(word in col.lower() 
                              for word in ['mental', 'health', 'treatment', 'anxiety', 'panic'])]
            if possible_targets:
                target_col = possible_targets[-1]  # Use the last one as it might be most relevant
            else:
                target_col = df.columns[-1]  # Use last column as fallback
        
        print(f"Using target column: {target_col}")
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != target_col:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Encode target variable if it's categorical
        if target_col in categorical_columns:
            le_target = LabelEncoder()
            df[target_col] = le_target.fit_transform(df[target_col])
            self.label_encoders['target'] = le_target
        
        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_columns = X.columns.tolist()
        print(f"Feature columns: {self.feature_columns}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, df
    
    def evaluate_models(self, X, y):
        """Evaluate all models and find the best one"""
        print("\nEvaluating models...")
        print("-" * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            
            model_scores[name] = {
                'accuracy': accuracy,
                'cv_score': cv_mean,
                'model': model
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, CV Score: {cv_mean:.4f}")
        
        # Find best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['accuracy'])
        self.best_model = model_scores[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {model_scores[best_model_name]['accuracy']:.4f}")
        
        return model_scores
    
    def save_model(self, model_path='best_mental_health_model.pkl'):
        """Save the best model and preprocessors"""
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='best_mental_health_model.pkl'):
        """Load the saved model and preprocessors"""
        model_data = joblib.load(model_path)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        print(f"Model loaded: {self.best_model_name}")
    
    def predict(self, input_data):
        """Make prediction on new data"""
        if self.best_model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Ensure input_data is a DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Make sure we only use the features that were used during training
        processed_data = input_data.copy()
        
        # Only keep columns that exist in feature_columns
        available_features = [col for col in self.feature_columns if col in processed_data.columns]
        missing_features = [col for col in self.feature_columns if col not in processed_data.columns]
        
        if missing_features:
            print(f"Warning: Missing features will be filled with 0: {missing_features}")
            # Add missing features with default values
            for col in missing_features:
                processed_data[col] = 0
        
        # Select only the training features in the correct order
        processed_data = processed_data[self.feature_columns]
        
        # Apply label encoders to categorical columns
        for col in processed_data.columns:
            if col in self.label_encoders and col != 'target':
                le = self.label_encoders[col]
                # Handle unseen categories by using the most frequent class or 0
                processed_data[col] = processed_data[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # Scale features
        processed_data_scaled = self.scaler.transform(processed_data)
        
        # Make prediction
        prediction = self.best_model.predict(processed_data_scaled)[0]
        prediction_proba = self.best_model.predict_proba(processed_data_scaled)[0]
        
        # Decode prediction if necessary
        if 'target' in self.label_encoders:
            prediction_label = self.label_encoders['target'].inverse_transform([prediction])[0]
        else:
            prediction_label = prediction
        
        confidence = max(prediction_proba)
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'probabilities': prediction_proba.tolist()
        }

def main():
    """Main function to train the model"""
    # Initialize predictor
    predictor = StudentMentalHealthPredictor()
    
    # Load and preprocess data
    # Note: Replace 'Student Mental Health.csv' with your actual file path
    try:
        X, y, df = predictor.load_and_preprocess_data('Student Mental Health.csv')
        
        # Evaluate models
        model_scores = predictor.evaluate_models(X, y)
        
        # Save the best model
        predictor.save_model()
        
        print("\nModel training completed successfully!")
        
    except FileNotFoundError:
        print("Dataset file not found. Please ensure 'Student Mental Health.csv' is in the current directory.")
        print("You can download it from: https://www.kaggle.com/datasets/shariful07/student-mental-health")

if __name__ == "__main__":
    main()