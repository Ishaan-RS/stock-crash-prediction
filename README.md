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

![WhatsApp Image 2025-04-17 at 11 44 01](https://github.com/user-attachments/assets/b90905fe-d9e7-4a41-b22c-543658a2b9ab)



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


Project Screenshots:

![image](https://github.com/user-attachments/assets/7821114c-08fd-4abc-a40e-e0d714f245c3)

![image](https://github.com/user-attachments/assets/f5179ef4-05ff-4efa-a912-b910766677ec)

![image](https://github.com/user-attachments/assets/1979cd90-e4ab-4dd3-afe9-0a856c8c2e7e)

![image](https://github.com/user-attachments/assets/f884b709-4306-45df-82c6-74545bc44a86)

![image](https://github.com/user-attachments/assets/33401885-32af-44a9-a4db-260c4adef6e5)

![image](https://github.com/user-attachments/assets/a1da4366-2aeb-49ef-91fe-4e36c8fb8bd6)

![image](https://github.com/user-attachments/assets/8a0dfee7-351b-4c9e-8e2c-ff0121d06689)

![image](https://github.com/user-attachments/assets/ab65f73c-d1a5-40d0-8092-4579fd951e86)

![image](https://github.com/user-attachments/assets/bac6bce1-83d5-49f0-97fd-efaef054311d)

![image](https://github.com/user-attachments/assets/0e118e75-be97-4d0d-a0bb-c2a36ed181b8)

![image](https://github.com/user-attachments/assets/c6a4a567-54cb-4761-8c1e-176da70edb27)

![image](https://github.com/user-attachments/assets/95ccbdd0-5326-49c5-9e80-9e25541580a2)

---

Made with passion for data, finance, and AI.

---
