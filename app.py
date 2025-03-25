import os
from flask import Flask, render_template, request
import qrcode
from io import BytesIO
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# ------------------------------------------
# Hugging Face Inference API for sentiment
# ------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
# Replace with your actual token
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def analyze_sentiment(text):
    """
    Calls Hugging Face DistilBERT model to get POSITIVE or NEGATIVE sentiment.
    Because the model sometimes returns nested lists, we handle that carefully.
    """
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    print("DEBUG response text:", response.text)  # For troubleshooting
    try:
        response.raise_for_status()
        result = response.json()
        print("DEBUG result:", result)

        # Check if result is a nested list: [[{label,score}, {label,score}]]
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            predictions = result[0]  # e.g. [{'label': 'NEGATIVE', 'score': 0.9997}, ...]
            best = max(predictions, key=lambda x: x["score"])
            label = best["label"]
            score = best["score"]
            return f"{label} (score: {score:.4f})"

        # Otherwise, if it's a simple list of dicts: [{label,score}]
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            label = result[0]["label"]
            score = result[0]["score"]
            return f"{label} (score: {score:.4f})"

        return "Unexpected response format"

    except Exception as e:
        print("Error calling Hugging Face API:", e)
        return "Error analyzing sentiment"


def get_label_only(text):
    """
    For counting POSITIVE/NEGATIVE tallies, we only need the label.
    If the full result is "POSITIVE (score: 0.9998)", we'll parse out "POSITIVE".
    """
    full_result = analyze_sentiment(text)
    if full_result.startswith("POSITIVE"):
        return "POSITIVE"
    elif full_result.startswith("NEGATIVE"):
        return "NEGATIVE"
    else:
        return "NEGATIVE"  # default fallback if there's an error or "Unexpected"


# ------------------------------------------
# Local QR Code Generation
# ------------------------------------------
def generate_qr_code_local(text):
    """
    Generates a standard QR code (PNG) for the given text/URL
    and returns a base64-encoded string suitable for <img> tags.
    """
    img = qrcode.make(text)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + base64_img


# ------------------------------------------
# In-memory storage for sentiment counts
# ------------------------------------------
feedback_storage = {
    "cleanliness": {"POSITIVE": 0, "NEGATIVE": 0},
    "time": {"POSITIVE": 0, "NEGATIVE": 0},
    "courtesy": {"POSITIVE": 0, "NEGATIVE": 0},
    "food_quality": {"POSITIVE": 0, "NEGATIVE": 0}
}


# ------------------------------------------
# Routes
# ------------------------------------------
@app.route('/')
def index():
    """
    Home page: generates a QR code linking to /survey
    """
    # survey_url = request.host_url + "survey"  # e.g. http://127.0.0.1:5000/survey
    survey_url = "http://10.10.4.21:5000/survey"
    qr_image_data = generate_qr_code_local(survey_url)
    return render_template('index.html', qr_image=qr_image_data, survey_url=survey_url)


@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """
    Shows a form with 4 text areas: cleanliness, time, courtesy, food_quality.
    When submitted, we do sentiment analysis on each category and store the tallies.
    """
    if request.method == 'POST':
        # Grab the text from each field
        cleanliness_text = request.form.get('cleanliness', '')
        time_text = request.form.get('time', '')
        courtesy_text = request.form.get('courtesy', '')
        food_text = request.form.get('food_quality', '')

        # For each category, if not empty, get the label (POSITIVE or NEGATIVE) and update feedback_storage
        cleanliness_label = "No feedback"
        time_label = "No feedback"
        courtesy_label = "No feedback"
        food_label = "No feedback"

        if cleanliness_text.strip():
            label = get_label_only(cleanliness_text)
            cleanliness_label = label
            feedback_storage["cleanliness"][label] += 1

        if time_text.strip():
            label = get_label_only(time_text)
            time_label = label
            feedback_storage["time"][label] += 1

        if courtesy_text.strip():
            label = get_label_only(courtesy_text)
            courtesy_label = label
            feedback_storage["courtesy"][label] += 1

        if food_text.strip():
            label = get_label_only(food_text)
            food_label = label
            feedback_storage["food_quality"][label] += 1

        # Pass them to the result page
        return render_template(
            'result.html',
            cleanliness=cleanliness_text,
            cleanliness_label=cleanliness_label,
            time=time_text,
            time_label=time_label,
            courtesy=courtesy_text,
            courtesy_label=courtesy_label,
            food_quality=food_text,
            food_label=food_label
        )

    # If GET, just show the form
    return render_template('survey.html')


@app.route('/overall')
def overall():
    """
    Shows how many POSITIVE vs. NEGATIVE submissions we have for each category
    """
    return render_template('overall.html', data=feedback_storage)


# ------------------------------------------
# 5) Run the App
# ------------------------------------------
# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
