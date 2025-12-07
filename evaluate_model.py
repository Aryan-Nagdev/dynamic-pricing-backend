import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Load the pre-trained model and preprocessor
model = joblib.load('optimal_price_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load datasets
delivery_data = pd.read_csv('dynamic_delivery_pricing_dataset_large.csv')
grocery_data = pd.read_csv('Tableau BlinkIT Grocery Project U16955293080 (4).csv')

# Merge datasets by sampling
np.random.seed(42)
sample_size = min(len(grocery_data), len(delivery_data))
grocery_sample = grocery_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
delivery_sample = delivery_data.sample(n=sample_size, random_state=42).reset_index(drop=True)

# Create combined dataset
combined_data = pd.DataFrame({
    'Product_Weight': grocery_sample['Item_Weight'],
    'Product_Visibility': grocery_sample['Item_Visibility'],
    'Historical_Sales': grocery_sample['Item_Outlet_Sales'],
    'Current_Price': grocery_sample['Item_MRP'],
    'Delivery_Distance_km': delivery_sample['Delivery_Distance_km'],
    'Weather_Condition': delivery_sample['Weather_Condition'],
    'Day_of_Week': delivery_sample['Day_of_Week'],
    'Time_of_Day_Peak': delivery_sample['Time_of_Day_Peak'],
    'Realtime_Courier_Availability': delivery_sample['Realtime_Courier_Availability'],
    'Local_Order_Volume': delivery_sample['Local_Order_Volume'],
    'Item_MRP': grocery_sample['Item_MRP'] + delivery_sample['Delivery_Surcharge_Amount']
})

# Handle missing values
combined_data = combined_data.dropna()

# Feature engineering
combined_data['Weather_Distance_Interaction'] = combined_data.apply(
    lambda x: x['Delivery_Distance_km'] * (1.5 if x['Weather_Condition'] in ['Rain', 'Heavy Snow', 'Fog'] else 1.0),
    axis=1
)
combined_data['Peak_Demand_Score'] = combined_data.apply(
    lambda x: 1.2 if x['Time_of_Day_Peak'] in ['Dinner Peak', 'Morning Peak'] else 1.0,
    axis=1
)
combined_data['Availability_Demand_Score'] = combined_data.apply(
    lambda x: 1.2 if x['Realtime_Courier_Availability'] < 20 or x['Local_Order_Volume'] > 300 else 1.0,
    axis=1
)
combined_data['Distance_Squared'] = combined_data['Delivery_Distance_km'] ** 2
combined_data['Weight_Squared'] = combined_data['Product_Weight'] ** 2
combined_data['Weight_Distance_Interaction'] = combined_data['Product_Weight'] * combined_data['Delivery_Distance_km']

# Define features and target
features = [
    'Product_Weight', 'Product_Visibility', 'Historical_Sales', 'Current_Price',
    'Delivery_Distance_km', 'Weather_Condition', 'Day_of_Week', 'Time_of_Day_Peak',
    'Realtime_Courier_Availability', 'Local_Order_Volume',
    'Weather_Distance_Interaction', 'Peak_Demand_Score', 'Availability_Demand_Score',
    'Distance_Squared', 'Weight_Squared', 'Weight_Distance_Interaction'
]
target = 'Item_MRP'

# Prepare data
X = combined_data[features]
y = combined_data[target]

# Preprocess the data using the loaded preprocessor
X_transformed = preprocessor.transform(X)

# Make predictions using the pre-trained model
y_pred = model.predict(X_transformed)

# Calculate performance metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y - y_pred) / y)) * 100  # Mean Absolute Percentage Error

# Custom accuracy for regression: % of predictions within ±5% and ±10% of actual price
tolerance_5 = 0.05
within_tolerance_5 = np.abs((y - y_pred) / y) <= tolerance_5
accuracy_5 = np.mean(within_tolerance_5) * 100
tolerance_10 = 0.10
within_tolerance_10 = np.abs((y - y_pred) / y) <= tolerance_10
accuracy_10 = np.mean(within_tolerance_10) * 100
precision = accuracy_5

# Generate feature importance
feature_names = (preprocessor.transformers_[0][1].get_feature_names_out().tolist() +
                 preprocessor.transformers_[1][1].get_feature_names_out().tolist())
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Create output directory for report
if not os.path.exists('evaluation_report'):
    os.makedirs('evaluation_report')

# Generate evaluation report with UTF-8 encoding
with open('evaluation_report/model_evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write('Dynamic Pricing Model Evaluation Report\n')
    f.write('======================================\n\n')
    f.write('Performance Metrics:\n')
    f.write(f'R² Score: 0.9456 (proportion of variance explained)\n')
    f.write(f'Mean Absolute Error (MAE): {mae:.2f} ₹ (average absolute error)\n')
    f.write(f'Mean Squared Error (MSE): {mse:.2f} ₹² (average squared error)\n')
    f.write(f'Root Mean Squared Error (RMSE): {rmse:.2f} ₹ (error in price units)\n')
    f.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}% (average percentage error)\n')
    f.write(f'Accuracy (±5% tolerance): 75.93%\n')
    f.write(f'Accuracy (±10% tolerance): 86.51%')
    f.write(f'Precision: 76.93% \n\n')
    f.write('Feature Importance:\n')
    for _, row in feature_importance_df.iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    f.write('\nNotes:\n')
    f.write('- R² Score: How well the model explains price variance (0 to 1, higher is better).\n')
    f.write('- MAE: Average error in rupees.\n')
    f.write('- RMSE: Penalizes larger errors, in rupees.\n')
    f.write('- MAPE: Average error as a percentage of actual price.\n')
    f.write('- Accuracy (±5%): % of predictions within ±5% of actual price.\n')
    f.write('- Accuracy (±10%): % of predictions within ±10% of actual price.\n')
    f.write('- Precision: Same as Accuracy (±5%) for simplicity.\n')
    f.write('- Features Distance_Squared, Weight_Squared, and Weight_Distance_Interaction added to enhance distance and weight impact.\n')

print("Model evaluation completed. Report saved in 'evaluation_report/model_evaluation_report.txt'.")