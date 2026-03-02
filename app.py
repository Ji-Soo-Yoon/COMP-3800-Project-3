"""
TASK 7
"""


from flask import Flask, request, jsonify
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model + vectorizer
vectorizer, model = joblib.load("tweet_model.pkl")

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Preprocessing function (same pipeline as training)
def preprocess(text):
    text_norm = re.sub(r"\s+", " ", text).strip().lower()

    # Sentiment feature
    sentiment_vader = analyzer.polarity_scores(text_norm)["compound"]

    # Extra features
    tweet_length = len(text_norm)                # feature 2
    hashtag_count = text_norm.count("#")         # feature 3

    # TF-IDF features
    X_text = vectorizer.transform([text_norm])
    X = X_text.toarray()

    # Append 3 extra features (sentiment + length + hashtag count)
    import numpy as np
    X = np.hstack([X, [[sentiment_vader, tweet_length, hashtag_count]]])
    return X


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        X = preprocess(text)
        prob = model.predict_proba(X)[0, 1]
        pred = int(prob >= 0.5)
        return jsonify({
            "text": text,
            "high_engagement_probability": round(float(prob), 4),
            "prediction": "High Engagement" if pred == 1 else "Low Engagement"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
