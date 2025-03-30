from flask import Flask, render_template, request
import qrcode
from io import BytesIO
import base64
from transformers import pipeline

app = Flask(__name__)

# ------------------------------------------
# 1) Local Sentiment Analysis using Transformers
# ------------------------------------------
# Load the sentiment-analysis pipeline locally.
# This uses the "distilbert-base-uncased-finetuned-sst-2-english" model.
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    """
    Runs local sentiment analysis on the given text.
    Returns a string like "POSITIVE (score: 0.9876)".
    """
    result = sentiment_pipeline(text)[0]  # e.g. {'label': 'POSITIVE', 'score': 0.9998}
    return f"{result['label']} (score: {result['score']:.4f})"

def get_label_only(text):
    """
    Returns just "POSITIVE" or "NEGATIVE" from the analyze_sentiment result.
    """
    full_result = analyze_sentiment(text)
    full_result_lower = full_result.lower()
    if "positive" in full_result_lower:
        return "POSITIVE"
    elif "negative" in full_result_lower:
        return "NEGATIVE"
    else:
        return "NEGATIVE"

# ------------------------------------------
# 2) Local QR Code Generation
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
# 3) In-memory Storage for Feedback Texts
# ------------------------------------------
feedback_storage = {
    "cleanliness": {"POSITIVE": [], "NEGATIVE": []},
    "time": {"POSITIVE": [], "NEGATIVE": []},
    "courtesy": {"POSITIVE": [], "NEGATIVE": []},
    "food_quality": {"POSITIVE": [], "NEGATIVE": []}
}

# ------------------------------------------
# 4) Routes
# ------------------------------------------
@app.route('/')
def index():
    """
    Home page: generates a QR code linking to /survey.
    Adjust the survey URL to your local IP so your phone can access it.
    """
    # Replace with your computer's actual local IP address.
    survey_url = "http://10.10.1.36:5000/survey"
    qr_image_data = generate_qr_code_local(survey_url)
    return render_template('index.html', qr_image=qr_image_data, survey_url=survey_url)

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """
    Displays a form with 4 text areas for feedback.
    The sentiment of each category is determined locally and feedback stored.
    """
    if request.method == 'POST':
        cleanliness_text = request.form.get('cleanliness', '')
        time_text = request.form.get('time', '')
        courtesy_text = request.form.get('courtesy', '')
        food_text = request.form.get('food_quality', '')

        cleanliness_label = "No feedback"
        time_label = "No feedback"
        courtesy_label = "No feedback"
        food_label = "No feedback"

        if cleanliness_text.strip():
            lbl = get_label_only(cleanliness_text)
            cleanliness_label = lbl
            feedback_storage["cleanliness"][lbl].append(cleanliness_text)

        if time_text.strip():
            lbl = get_label_only(time_text)
            time_label = lbl
            feedback_storage["time"][lbl].append(time_text)

        if courtesy_text.strip():
            lbl = get_label_only(courtesy_text)
            courtesy_label = lbl
            feedback_storage["courtesy"][lbl].append(courtesy_text)

        if food_text.strip():
            lbl = get_label_only(food_text)
            food_label = lbl
            feedback_storage["food_quality"][lbl].append(food_text)

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
    return render_template('survey.html')

@app.route('/overall')
def overall():
    """
    Overall results page: displays, for each category, a table with two columns
    (Positive and Negative) listing the feedback texts.
    """
    return render_template('overall.html', data=feedback_storage)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
