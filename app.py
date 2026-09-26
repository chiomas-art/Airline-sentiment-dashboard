import streamlit as st
import pandas as pd
import re
import glob
import kagglehub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Airline Sentiment Dashboard", page_icon="🟣", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #EDE7F6 0%, #D1C4E9 100%); }
section[data-testid="stSidebar"] { background-color: #6A3FA0; }
section[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stMetric"] { background-color: #6A3FA0; padding: 15px; border-radius: 12px; color: white; }
[data-testid="stMetricLabel"] { color: #EDE7F6 !important; }
h1, h2, h3 { color: #4A1D8C; }
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { background-color: #6A3FA0; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🟣 US Airline Sentiment Classification Dashboard")
st.caption("Automated customer-feedback triage — loaded live from Kaggle")

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")
    csv_files = glob.glob(f"{path}/**/*.csv", recursive=True)
    csv_file = [f for f in csv_files if 'tweet' in f.lower()][0] if any('tweet' in f.lower() for f in csv_files) else csv_files[0]
    df = pd.read_csv(csv_file)
    return df[['text', 'airline', 'airline_sentiment']].dropna(subset=['text', 'airline_sentiment'])

def clean_tweet(t):
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'http\S+', '', t)
    t = re.sub(r'[^a-zA-Z\s]', ' ', t)
    return t.lower().strip()

with st.spinner("Downloading dataset from Kaggle (first run only)..."):
    df = load_data()
df['clean_text'] = df['text'].apply(clean_tweet)

@st.cache_resource
def train_models(df):
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_text'], df['airline_sentiment'], test_size=0.2, random_state=42, stratify=df['airline_sentiment']
    )
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=2)
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    logreg = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    logreg.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, logreg.predict(X_test))

    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))

    return tfidf, logreg, rf, acc_lr, acc_rf, y_test, logreg.predict(X_test)

tfidf, logreg, rf, acc_lr, acc_rf, y_test, pred_lr = train_models(df)

tab1, tab2, tab3 = st.tabs(["📊 Past — EDA", "🎯 Present — Model Performance", "🔮 Future — Live Predictor"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tweets Analyzed", f"{len(df):,}")
    col2.metric("Airlines Tracked", df['airline'].nunique())
    col3.metric("Negative Share", f"{(df['airline_sentiment'] == 'negative').mean()*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df['airline_sentiment'].value_counts().plot(kind='bar', ax=axes[0], color=['#6A3FA0', '#9575CD', '#D1C4E9'])
    axes[0].set_title('Overall Sentiment Distribution')
    axes[0].tick_params(axis='x', rotation=0)
    sentiment_by_airline = pd.crosstab(df['airline'], df['airline_sentiment'], normalize='index')
    sentiment_by_airline.plot(kind='bar', stacked=True, ax=axes[1], color=['#6A3FA0', '#9575CD', '#E1D5F5'])
    axes[1].set_title('Sentiment Share by Airline')
    axes[1].legend(title='Sentiment', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression Accuracy", f"{acc_lr*100:.1f}%")
    col2.metric("Random Forest Accuracy", f"{acc_rf*100:.1f}%")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred_lr, labels=logreg.classes_), display_labels=logreg.classes_).plot(ax=ax2, cmap='Purples')
    st.pyplot(fig2)
    st.text(classification_report(y_test, pred_lr))

with tab3:
    st.subheader("Live Tweet Sentiment Triage")
    user_input = st.text_area("Enter a customer tweet to classify:",
                               "The flight was severely delayed and customer service was unhelpful.")
    if st.button("Predict Sentiment"):
        cleaned = clean_tweet(user_input)
        vec = tfidf.transform([cleaned])
        prediction = logreg.predict(vec)[0]
        confidence = max(logreg.predict_proba(vec)[0]) * 100
        if prediction == 'negative':
            st.error(f"Predicted: **{prediction.upper()}** ({confidence:.0f}% confidence)")
        elif prediction == 'positive':
            st.success(f"Predicted: **{prediction.upper()}** ({confidence:.0f}% confidence)")
        else:
            st.info(f"Predicted: **{prediction.upper()}** ({confidence:.0f}% confidence)")