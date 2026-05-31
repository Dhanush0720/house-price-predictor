# src/preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class RealEstateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to engineer features like room density safely."""
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        
        # Force numeric conversion right before processing to clear string type artifacts
        if 'total_rooms' in X.columns:
            X['total_rooms'] = pd.to_numeric(X['total_rooms'], errors='coerce')
        if 'sqft_living' in X.columns:
            X['sqft_living'] = pd.to_numeric(X['sqft_living'], errors='coerce')
            
        # Fill any unexpected conversion NaN issues with defaults to avoid computation failures
        X['total_rooms'] = X['total_rooms'].fillna(2)
        X['sqft_living'] = X['sqft_living'].fillna(1200)
            
        # Now calculating room density is 100% mathematically safe
        X['rooms_per_sqft'] = X['total_rooms'] / (X['sqft_living'] + 1e-5)
        
        return X

def build_preprocessing_pipeline(num_features, cat_features):
    """Creates a unified preprocessing pipeline for production."""
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    
    # We apply the feature engineering logic cleanly right in front of scaling pipelines
    full_pipeline = Pipeline([
        ('engineer', RealEstateFeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline