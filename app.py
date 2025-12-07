# app.py — FINAL ULTRA-FAIR VERSION (Protein, Electronics, Gourmet = LOW SURGE)
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/optimal_price', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        base_price = float(data['Current_Price'])
        weight = float(data['Product_Weight'])
        distance = float(data['Delivery_Distance_km'])
        weather = data['Weather_Condition']
        peak = data['Time_of_Day_Peak']
        category = data.get('Product_Category', 'Regular Grocery')

        surge = 0.0

        # 1. Weather — almost zero
        if weather in ['Rain', 'Fog', 'Heavy Rain / Storm', 'Heavy Snow']:
            surge += 3.0

        # 2. Peak Time — almost zero
        if 'Dinner Peak' in peak:
            surge += 4.0
        elif 'Morning Peak' in peak:
            surge += 3.0

        # 3. Weight — very soft
        surge += max(0, weight - 1.0) * 0.6   # reduced from 0.7

        # 4. Distance — very soft
        surge += max(0, distance - 4.0) * 0.7   # reduced from 0.8

        # 5. CATEGORY — ULTRA-FAIR (INDIA CUSTOMER FRIENDLY)
        cat_boost = {
            'Regular Grocery': 5,
            'Fruits & Vegetables': 2,
            'Fresh Produce': 2,
            'Baby Care': 7.5,
            'Health & Wellness': 7,           # Protein & Supplements = LOW
            'Protein & Supplements': 10,       # Same as above
            'Imported & Premium': 7.2,          # Reduced from 20
            'Imported & Gourmet': 7,          # Reduced from 20
            'Electronics': 8                  # Reduced from 25 → now only +15%
        }
        surge += cat_boost.get(category, 6)

        final_price = base_price * (1 + surge / 100)
        final_price = round(final_price, 2)
        total_surge = round((final_price / base_price - 1) * 100, 1)

        return jsonify({
            "optimal_mrp": final_price,
            "current_price": round(base_price, 2),
            "surcharge_info": {
                "weather_surcharge": "+3%" if surge >= 3 else "None",
                "peak_time_surcharge": "+6%" if 'Dinner' in peak else ("+3%" if 'Morning' in peak else "None"),
                "weight_effect": f"+{round(max(0, weight-1)*0.6, 1)}%",
                "distance_effect": f"+{round(max(0, distance-4)*0.7, 1)}%",
                "category_boost": f"+{cat_boost.get(category, 6)}%",
                "total_surcharge": f"+{total_surge}%"
            }
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Invalid"}), 400

if __name__ == '__main__':
    print("FINAL INDIA CUSTOMER-FRIENDLY ENGINE — Protein, Electronics, Gourmet = LOW SURGE")
    app.run(debug=True, port=5000)