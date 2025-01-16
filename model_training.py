import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib  # For saving the model

def load_arff_data(file_path):
    data, meta = arff.loadarff(file_path)
    arff_df = pd.DataFrame(data)
    
    # Label encoding for categorical columns
    label_encoder = LabelEncoder()
    for column in arff_df.columns:
        if arff_df[column].dtype == 'object':  # Encode categorical data
            arff_df[column] = label_encoder.fit_transform(arff_df[column])

    # Select features and target
    arff_features = arff_df.drop(columns=['Performance'])
    arff_target = arff_df['Performance']
    
    return arff_features, arff_target

def preprocess_synthetic_dataset(synthetic_data):
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Ethnicity', 'ParentalEducation']
    
    for column in categorical_columns:
        synthetic_data[column] = label_encoder.fit_transform(synthetic_data[column].astype(str))
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    synthetic_data_imputed = pd.DataFrame(imputer.fit_transform(synthetic_data), columns=synthetic_data.columns)
    
    features = synthetic_data_imputed.drop(columns=['GradeClass', 'StudentID'])
    target = synthetic_data_imputed['GradeClass']
    
    return features, target

def merge_datasets(features1, target1, features2, target2):
    # Merge the datasets
    combined_features = pd.concat([features1, features2], axis=0)
    combined_target = pd.concat([target1, target2], axis=0)
    
    return combined_features, combined_target

def create_model_pipeline(categorical_columns):
    # Imputation for missing values
    imputer = SimpleImputer(strategy='mean')
    
    # Preprocessor for categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', imputer, ['age', 'failures', 'studytime']),  # Specify numerical columns for imputation
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # One-Hot Encoding for categorical columns
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return model

def train_combined_model(combined_features, combined_target, categorical_columns):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_target, test_size=0.2, random_state=42)
    
    # Create the model pipeline
    model = create_model_pipeline(categorical_columns)
    model.fit(X_train, y_train)
    
    # Print performance metrics (optional)
    print(f"Train Score: {model.score(X_train, y_train)}")
    print(f"Test Score: {model.score(X_test, y_test)}")
    
    return model

def save_model(model, model_path='student_performance_model.pkl'):
    joblib.dump(model, model_path)

def load_model(model_path='student_performance_model.pkl'):
    return joblib.load(model_path)
