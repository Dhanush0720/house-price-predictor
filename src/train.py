# src/train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from preprocessing import build_preprocessing_pipeline

def clean_and_align_columns(df):
    """
    Cleans raw real-world columns and standardizes them 
    to match the feature names expected by app.py.
    """
    df = df.copy()
    
    # 📝 STEP A: Map raw CSV headers to our system headers
    mapping = {
        'total_sqft': 'sqft_living',
        'size': 'total_rooms',
        'location': 'neighborhood',
        'price': 'price'
    }
    df = df.rename(columns=mapping)
    
    # Verify if the columns exist after mapping to prevent KeyErrors
    required_cols = ['sqft_living', 'total_rooms', 'neighborhood', 'price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise KeyError(
            f"\n❌ Column Mapping Error! Missing required columns: {missing_cols}.\n"
            f"Actual columns in your CSV are: {list(df.columns)}\n"
            f"Please modify the 'mapping' dictionary in src/train.py to match your CSV headers."
        )
    
    # 📝 STEP B: Clean BHK / Room text into pure numeric integers (e.g., "3 BHK" -> 3)
    if df['total_rooms'].dtype == 'object':
        df['total_rooms'] = df['total_rooms'].astype(str).str.extract(r'(\d+)').astype(float)
        
    # 📝 STEP C: Clean Square Footage text brackets (e.g., "1200 - 1400" -> 1300)
    def convert_range_to_num(x):
        try:
            if '-' in str(x):
                split_val = str(x).split('-')
                return (float(split_val[0].strip()) + float(split_val[1].strip())) / 2
            return float(x)
        except:
            return None

    if df['sqft_living'].dtype == 'object':
        df['sqft_living'] = df['sqft_living'].apply(convert_range_to_num)

    # 📝 STEP D: Convert target price scales from Lakhs to Raw Indian Rupees
    df['price'] = df['price'] * 100000
    
    # Drop rows missing vital fields post-cleanup
    df = df.dropna(subset=required_cols)
    
    return df

def train_model():
    csv_path = os.path.join('data', 'raw', 'indian_housing.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌ Missing dataset file! Please place your downloaded real estate CSV at: '{csv_path}'"
        )
        
    print(f"📦 Ingesting real-world dataset from: {csv_path}...")
    raw_df = pd.read_csv(csv_path)
    
    # Clean and parse structural features
    data = clean_and_align_columns(raw_df)
    print(f"🧹 Data pipeline parsing complete. Loaded {data.shape[0]} valid property matrices.")
    
    # Assign inputs and outputs
    X = data[['sqft_living', 'total_rooms', 'neighborhood']]
    y = data['price']
    
    num_features = ['sqft_living', 'total_rooms']
    cat_features = ['neighborhood']
    
    # 2. Split Data cleanly into Training and Validation profiles
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Build & Fit Production ML Pipeline Object
    print("⚙️ Processing features and training production model pipeline...")
    pipeline = build_preprocessing_pipeline(num_features, cat_features)
    
    model_pipeline = Pipeline([
        ('preprocessing', pipeline),
        ('regressor', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    # 4. Evaluate Pipeline Accuracy Metrics
    predictions = model_pipeline.predict(X_test)
    print(f"✅ Training Complete!")
    print(f"📊 Model R² Accuracy Score: {r2_score(y_test, predictions):.2f}")
    print(f"📊 Model Mean Absolute Error (MAE): ₹{mean_absolute_error(y_test, predictions):,.2f}")
    
    # 5. Save Finished Production Artifact
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/model.joblib')
    print("💾 Saved production model pipeline binary to 'models/model.joblib'")

if __name__ == "__main__":
    train_model()