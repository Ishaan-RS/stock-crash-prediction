from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
import io
import base64
import os
import google.generativeai as genai

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# === Set up LangChain Gemini LLM ===
genai.configure(api_key="YOUR_SECRET_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_followup_insights(result):
    prompt = f"""
You are a financial analyst assistant. Given the following stock analysis results. Don't mention name, stock name. The currency unit is in indian rupees:

- Highest Close: ${result['highest']}
- Lowest Close: ${result['lowest']}
- Volatility: ${result['volatility']}
- Percentage Change: ${result['pct_change']}%
- Market Trend: ${result['trend']}
- Pattern: ${result['pattern']}

Please provide a brief yet insightful follow-up analysis that explains what this means for an average investor. Keep it friendly and non-technical.
"""
    response = model.generate_content(prompt)
    print(response.text)
    return response.text

def score(y):
    return len(y[y > 0.32])

def clean_text(text):
    # Handle NaN values if any
    if pd.isna(text):
        return ''
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    # Remove user mentions and hashtags (but keep the words)
    text = re.sub(r'@\w+|\#', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())

    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load everything
lstm_model = load_model('stockmodel.h5')  # for historical data
sentiment_model = joblib.load('logistic.joblib')  # e.g., LogisticRegression
tfidf_vectorizer = joblib.load('vectorizer.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Get uploaded CSV file
            csv_file = request.files['market_csv']
            df = pd.read_csv(csv_file)

            if df.shape != (15, 6):
                return "CSV must contain exactly 15 rows and 6 columns of numerical data."

            # 2. Scale and reshape the market data
            scaled_data = scaler.transform(df.values)
            X_seq = scaled_data.reshape(15, 1, 6)

            # 3. LSTM model prediction
            lstm_preds = lstm_model.predict(X_seq).flatten()
            count = score(lstm_preds)
            lstm_confidence = int(count >= 8)

            # 4. Text-based sentiment prediction
            user_text = request.form['current_affairs']
            text_vector = tfidf_vectorizer.transform([clean_text(user_text)])
            sentiment_confidence = sentiment_model.predict(text_vector)[0]

            # 5. Combine predictions
            return render_template("predict.html",
                                    prediction=lstm_confidence,
                                    lstm_conf=lstm_confidence * 100,
                                   sentiment_conf=sentiment_confidence * 100,
                                   user_text=user_text)

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("predict.html")


@app.route('/insights', methods=['GET', 'POST'])
def insights():
    result = None
    plot_data = None
    error = None
    followup = None

    if request.method == 'POST':
        file = request.files['file']
        try:
            df = pd.read_csv(file)
            required_cols = {'open', 'high', 'low', 'close', 'hour', 'minute'}
            if not required_cols.issubset(df.columns.str.lower()):
                error = f"CSV must contain: {', '.join(required_cols)}"
                return render_template("index.html", error=error)

            df.columns = df.columns.str.lower()
            if len(df) < 15:
                return render_template('index.html', error="Please upload at least 15 rows.")

            df = df.tail(15).reset_index(drop=True)

            highest = df['close'].max()
            lowest = df['close'].min()
            volatility = df['close'].std()
            price_change = df['close'].iloc[-1] - df['close'].iloc[0]
            pct_change = (price_change / df['close'].iloc[0]) * 100
            trend = "Bullish 📈" if price_change > 0 else "Bearish 📉"

            if pct_change > 0.2 and volatility < 4:
                pattern = "Mild Bullish with Low Volatility"
            elif pct_change < -0.2 and volatility > 5:
                pattern = "Possible Downtrend with High Volatility"
            else:
                pattern = "Neutral/Sideways Market"

            fig, ax = plt.subplots()
            ax.plot(df['close'], marker='o', linestyle='-')
            ax.set_title("Stock Price Trend")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Close Price")
            ax.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            result = {
                'highest': round(highest, 2),
                'lowest': round(lowest, 2),
                'volatility': round(volatility, 2),
                'pct_change': round(pct_change, 2),
                'trend': trend,
                'pattern': pattern
            }

            # LangChain-generated insights
            followup = generate_followup_insights(result)

        except Exception as e:
            error = f"Error processing file: {e}"

    return render_template("insights.html", result=result, plot_url=plot_data, error=error, followup=followup)

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

if __name__ == '__main__':
    app.run(debug=True)
