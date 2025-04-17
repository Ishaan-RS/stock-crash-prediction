# Stock Market Crash Predictor

A full-stack Flask web application to predict potential stock market crashes using sentiment analysis and historical market data. This project combines AI-driven insights, data visualization, and user-friendly UI/UX to deliver a comprehensive crash prediction tool.

## Features

- *Crash Prediction from Tweet Text*: Predict potential crashes using a sentiment analysis model on user-inputted tweets or headlines.
- *Crash Prediction from Market Data*: Upload historical stock market data (CSV) and receive crash likelihood predictions using ML models.
- *GenAI Insights Page*: Get LangChain-powered explanations for crash scenarios based on input or context.
- *Power BI Dashboard Embed*: Visualize market trends, model outputs, and key financial indicators using Power BI.
- *Modern UI/UX*: Fully responsive design using Tailwind CSS and JavaScript. Includes themed navigation, sidebar, landing page, and more.

## Tech Stack

- *Frontend*: HTML, Tailwind CSS, JavaScript
- *Backend*: Python, Flask
- *AI/ML*: Scikit-learn, TFIDF, Sentiment Analysis, Feature Scaling
- *Visualization*: Power BI (embedded dashboard)
- *Google Gemini*: For generating contextual GenAI insights (API)

## Project Structure

project-root/ │ ├── app.py                     # Main Flask backend ├── templates/ │   ├── index.html             # Landing/Home page │   ├── crash_from_tweet.html  # Text-based crash prediction │   ├── crash_from_data.html   # CSV-based crash prediction │   ├── genai.html             # LangChain insights │   └── dashboard.html         # Power BI dashboard page │ ├── static/ │   ├── css/                   # Tailwind and custom styles │   ├── js/                    # JS scripts │   └── assets/                # Images, icons, etc. │ ├── models/ │   ├── tfidf_vectorizer.pkl   # Pretrained TFIDF vectorizer │   ├── sentiment_model.pkl    # Sentiment-based crash model │   └── market_model.pkl       # Market data-based crash model │ ├── uploads/                   # Uploaded CSVs ├── requirements.txt           # Dependencies └── README.md                  # Project documentation

## Setup Instructions

1. *Clone the Repository*

git clone https://github.com/yourusername/stock-crash-predictor.git
cd stock-crash-predictor

2. Create a Virtual Environment



python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies



pip install -r requirements.txt

4. Run the Flask App



python app.py

Then visit http://127.0.0.1:5000/ in your browser.

How It Works

Text-Based Prediction:

User inputs a tweet or news headline.

The app vectorizes the text using TFIDF and predicts using a sentiment-trained classifier.


Data-Based Prediction:

User uploads CSV containing historical market indicators.

Features are scaled and passed to a model trained to predict crash probability.


GenAI Insights:

Generates explanations for why a crash might occur based on uploaded data or user prompts.

Uses GoogleGemini (mocked for now or with optional integration).



Example Tweet for Testing

> "BREAKING: Global markets plummet as inflation hits record highs. Dow is down 1200 points. Analysts warn this could be the start of a major crash."



---

Made with passion for data, finance, and AI.

---
