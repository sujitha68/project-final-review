import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset'
data = pd.read_csv('cleaned_data.csv')

# Define columns to log transform and scale
columns_to_transform = ['Area', 'Production', 'Fertilizer', 'Pesticide']
columns_to_scale = ['Crop_Year', 'Annual_Rainfall', 'Area', 'Production', 'Fertilizer', 'Pesticide']
categorical_features = ['Crop', 'State', 'Season']

# Separate features and target variable
target_column = 'Yield'
X = data.drop(target_column, axis=1)
y = data[target_column]

# Log transformer function
log_transformer = FunctionTransformer(np.log1p, validate=True)

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('log_and_scale', Pipeline(steps=[
            ('log', log_transformer),
            ('scale', StandardScaler())
        ]), columns_to_transform),
        ('scale_only', StandardScaler(), [col for col in columns_to_scale if col not in columns_to_transform]),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ]
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        colsample_bytree=0.9,
        gamma=0.1,
        learning_rate=0.2,
        max_depth=5,  # Reduced depth
        n_estimators=70,  # Reduced number of estimators
        reg_alpha=1,  # Increased alpha
        reg_lambda=2,  # Increased lambda
        subsample=0.7,
        
    ))
])


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Fit the model
pipeline.fit(X_train, y_train)

# Predict on the test set to evaluate the model
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}, R2 Score: {r2:.2f}")
print("XgBoost Regressor Test Set MSE:", mse)


# Save the trained pipeline to a pickle file
with open('pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("Pipeline saved to pipeline.pkl")


