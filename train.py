# train.py — FINAL VERSION (ALL PARAMETERS AFFECT PRICE)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

np.random.seed(42)

print("Loading data...")
delivery = pd.read_csv('dynamic_delivery_pricing_dataset_large.csv')
grocery = pd.read_csv('Tableau BlinkIT Grocery Project U16955293080 (4).csv')

sample = min(len(grocery), len(delivery))
g = grocery.sample(sample, random_state=42).reset_index(drop=True)
d = delivery.sample(sample, random_state=42).reset_index(drop=True)

df = pd.DataFrame({
    'Product_Weight': g['Item_Weight'].fillna(2.0),
    'Current_Price': g['Item_MRP'],
    'Delivery_Distance_km': d['Delivery_Distance_km'],
    'Weather_Condition': d['Weather_Condition'],
    'Time_of_Day_Peak': d['Time_of_Day_Peak'],
    'Product_Category': g['Item_Type']
})

# === Make category sensitive ===
category_surge = {
    'Dairy': 5,
    'Soft Drinks': 3,
    'Meat': 12,
    'Fruits and Vegetables': 8,
    'Snack Foods': 4,
    'Household': 6,
    'Frozen Foods': 7,
    'Breakfast': 5,
    'Health and Hygiene': 15,
    'Hard Drinks': 20,
    'Canned': 6,
    'Baking Goods': 4,
    'Others': 10,
    'Starchy Foods': 5,
    'Sea Food': 18
}

df['Category_Surge'] = df['Product_Category'].map(category_surge).fillna(5)

# === Realistic target ===
def optimal_price(row):
    base = row['Current_Price']
    surge = row['Category_Surge']

    if row['Weather_Condition'] in ['Rain', 'Heavy Snow', 'Fog']:
        surge += 10

    if row['Time_of_Day_Peak'] == 'Dinner Peak (High Demand)':
        surge += 20
    elif 'Peak' in row['Time_of_Day_Peak']:
        surge += 10

    surge += max(0, row['Product_Weight'] - 1.0) * 1.2
    surge += max(0, row['Delivery_Distance_km'] - 4.0) * 1.5

    return base * (1 + surge / 100)

df['Optimal_Price'] = df.apply(optimal_price, axis=1)

features = ['Current_Price', 'Product_Weight', 'Delivery_Distance_km', 'Weather_Condition', 'Time_of_Day_Peak', 'Product_Category']

X = df[features]
y = df['Optimal_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer([
    ('num', StandardScaler(), ['Current_Price', 'Product_Weight', 'Delivery_Distance_km']),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
     ['Weather_Condition', 'Time_of_Day_Peak', 'Product_Category'])
])

model = Pipeline([
    ('prep', prep),
    ('xgb', xgb.XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.07, random_state=42))
])

print("Training...")
model.fit(X_train, y_train)
print(f"R²: {model.score(X_test, y_test):.4f}")

joblib.dump(model.named_steps['xgb'], 'optimal_price_model.pkl')
joblib.dump(model.named_steps['prep'], 'preprocessor.pkl')
print("Model saved — all parameters affect price!") 