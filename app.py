from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter, deque
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

app = Flask(__name__)
CORS(app)

# Data storage
user_data = deque(maxlen=100)  # Store up to 100 recent draws
number_frequency = Counter()    # Track frequency of numbers

# Define color mapping for numbers (1-49)
COLOR_MAPPING = {
    1: "Red", 4: "Red", 7: "Red", 10: "Red", 13: "Red", 16: "Red", 19: "Red", 22: "Red", 25: "Red", 28: "Red", 31: "Red", 34: "Red", 37: "Red", 40: "Red", 43: "Red", 46: "Red",
    2: "Blue", 5: "Blue", 8: "Blue", 11: "Blue", 14: "Blue", 17: "Blue", 20: "Blue", 23: "Blue", 26: "Blue", 29: "Blue", 32: "Blue", 35: "Blue", 38: "Blue", 41: "Blue", 44: "Blue",
    3: "Green", 6: "Green", 9: "Green", 12: "Green", 15: "Green", 18: "Green", 21: "Green", 24: "Green", 27: "Green", 30: "Green", 33: "Green", 36: "Green", 39: "Green", 42: "Green", 45: "Green",
    49: "Yellow"
}

@app.route('/predict', methods=['GET'])
def predict():
    # Get the prediction method
    method = request.args.get('method', 'frequency')

    if method == 'frequency':
        number_prediction = predict_least_likely_numbers()
        color_prediction = predict_color_with_frequency()
    elif method == 'moving_average':
        number_prediction = predict_moving_average()
        color_prediction = predict_color_with_frequency()  # Using same color prediction logic
    elif method == 'weighted_recency':
        number_prediction = predict_weighted_recency()
        color_prediction = predict_color_with_frequency()
    elif method == 'time_decay':
        number_prediction = predict_time_decay()
        color_prediction = predict_color_with_frequency()
    elif method == 'ml':
        number_prediction = predict_ml()
        color_prediction = predict_color_with_frequency()
    else:
        number_prediction = []
        color_prediction = "Unknown"

    return jsonify({
        "least_likely_numbers": number_prediction,
        "predicted_color": color_prediction
    })

# Frequency-Based Prediction for Numbers (Least Likely)
def predict_least_likely_numbers():
    number_count = Counter()
    for draw in user_data:
        number_count.update(draw)
    # Ensure 4 numbers are returned
    least_likely_numbers = [num for num, count in number_count.most_common()[-4:]]
    return least_likely_numbers

# Moving Average Prediction
def predict_moving_average():
    window_size = 10
    recent_draws = list(user_data)[-window_size:]
    avg = np.mean(recent_draws, axis=0)
    return avg.astype(int).tolist()[:4]  # Ensure it returns exactly 4 numbers

# Weighted Recency Prediction
def predict_weighted_recency():
    weights = np.exp(np.arange(len(user_data)) * -0.1)  # Exponentially weighted
    weighted_counts = Counter()

    for i, draw in enumerate(user_data):
        for num in draw:
            weighted_counts[num] += weights[i]

    # Ensure 4 numbers are returned
    least_likely_numbers = [num for num, count in weighted_counts.most_common()[-4:]]
    return least_likely_numbers

# Time Decay Prediction (Decaying Window)
def predict_time_decay():
    decay_factor = 0.9
    time_decay_counts = Counter()

    for i, draw in enumerate(user_data):
        decay_weight = decay_factor ** (len(user_data) - i - 1)
        for num in draw:
            time_decay_counts[num] += decay_weight

    # Ensure 4 numbers are returned
    least_likely_numbers = [num for num, count in time_decay_counts.most_common()[-4:]]
    return least_likely_numbers

# Machine Learning (Random Forest) Model Prediction
def predict_ml():
    if len(user_data) < 10:  # Need at least 10 draws to train the model
        return predict_least_likely_numbers()

    # Prepare data for training
    X = []
    y = []
    for i in range(len(user_data) - 1):
        X.append(user_data[i])
        y.append(user_data[i + 1])  # Predict the next draw

    X = np.array(X)
    y = np.array(y)

    # Train RandomForest
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Predict the next draw and ensure 4 numbers are returned
    prediction = clf.predict([user_data[-1]])[0]
    return prediction.tolist()[:4]

# Color Prediction Logic (using frequency)
def predict_color_with_frequency():
    color_count = Counter()
    for draw in user_data:
        for num in draw:
            color = COLOR_MAPPING.get(num, "Unknown")
            color_count[color] += 1

    likely_colors = [color for color, count in color_count.items() if count >= 2]
    if not likely_colors:
        return "Unknown"

    most_likely_color = color_count.most_common(1)[0][0]
    return most_likely_color

@app.route('/submit', methods=['POST'])
def submit_draw():
    draw = request.json.get('draw')
    if not draw or len(draw) != 6 or not all(1 <= num <= 49 for num in draw):
        return jsonify({"error": "Invalid draw"}), 400

    user_data.append(draw)
    return jsonify({"message": "Draw submitted successfully"}), 200

@app.route('/clear', methods=['POST'])
def clear_data():
    user_data.clear()
    return jsonify({"message": "Data cleared successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)
