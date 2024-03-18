import pandas as pd

# Load the dataset
file_path = 'C:/Users/Vineetha Burra/Desktop/Machine_Learning/Midterm_Project/summer-products.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()
# Summarize the dataset for missing values and data types
data_summary = data.describe(include='all').transpose()
missing_values = data.isnull().sum()
data_types = data.dtypes

# Prepare a summary table
summary_table = pd.DataFrame({
    "Data Type": data_types,
    "Missing Values": missing_values,
    "Unique Values": data.nunique()
})

summary_table
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Handling missing values
# For numerical columns with missing values, fill with the median
numerical_columns_with_nans = ['rating_five_count', 'rating_four_count', 'rating_three_count',
                               'rating_two_count', 'rating_one_count']
for col in numerical_columns_with_nans:
    data[col] = data[col].fillna(data[col].median())

# For categorical columns with missing values, we'll fill with a placeholder value 'unknown'
categorical_columns_with_nans = ['product_color', 'product_variation_size_id', 'origin_country']
for col in categorical_columns_with_nans:
    data[col] = data[col].fillna('unknown')

# Dropping columns that won't be useful in predicting consumer demand
columns_to_drop = ['currency_buyer', 'theme', 'crawl_month', 'product_url', 'product_picture', import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Simulating additional features for current fashion trends and social media sentiment
# Assuming a normalized scale from 0 to 1, where 1 indicates high relevance or positive sentiment
np.random.seed(42)  # For reproducibility
data['fashion_trend_score'] = np.random.rand(data.shape[0])
data['social_media_sentiment'] = np.random.rand(data.shape[0])

# Selecting features including the simulated ones and the target variable 'units_sold'
features = ['rating_count', 'rating_five_count', 'rating_four_count',
            'rating_three_count', 'rating_two_count', 'rating_one_count',
            'merchant_rating_count', 'merchant_has_profile_picture',
            'product_variation_inventory', 'merchant_rating']
target = 'units_sold'

# Preparing the features and target variable
X = data[features]
y = data[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                   'product_id', 'merchant_profile_picture', 'has_urgency_banner', 'urgency_text']
data_cleaned = data.drop(columns=columns_to_drop)

# Encoding categorical variables
label_encoder = LabelEncoder()
categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col].astype(str))

# Display the first few rows of the cleaned dataset
data_cleaned.head()
import xgboost as xgb

# Convert the dataset into an optimized data structure called DMatrix that XGBoost supports
dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
dtest_xgb = xgb.DMatrix(X_test, label=y_test)

# Define XGBoost model parameters
params_xgb = {
    'objective': 'reg:squarederror',  # Objective for regression tasks
    'max_depth': 5,  # Depth of the trees
    'eta': 0.1,  # Learning rate
    'subsample': 0.7,  # Portion of data to grow trees
    'eval_metric': 'rmse'  # Evaluation metric
}

# Train the model
num_rounds = 100
bst_xgb = xgb.train(params_xgb, dtrain_xgb, num_rounds)

# Predicting
y_pred_xgb = bst_xgb.predict(dtest_xgb)

# Evaluation
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost MSE:", mse_xgb)
print("XGBoost R^2:", r2_xgb)

import lightgbm as lgb

# Prepare the dataset
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Define LightGBM model parameters
params_lgb = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

# Train the model without specifying 'verbose_eval'
num_boost_round = 100
bst_lgb = lgb.train(params_lgb,
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_train, lgb_test])

# Predicting
y_pred_lgb = bst_lgb.predict(X_test, num_iteration=bst_lgb.best_iteration)

# Evaluation
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)
print("LightGBM MSE:", mse_lgb)
print("LightGBM R^2:", r2_lgb)
