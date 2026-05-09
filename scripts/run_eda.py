import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sns.set(style="whitegrid")

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "raw", "raw_analyst_ratings.csv")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

def main():
    news = pd.read_csv(DATA_PATH)
    news['date'] = pd.to_datetime(news['date'], errors='coerce')

    # Plot 1: headline length distribution
    news['headline_len'] = news['headline'].astype(str).str.len()
    plt.figure(figsize=(8,4))
    sns.histplot(news['headline_len'].dropna(), bins=50)
    plt.title('Headline length distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'headline_length.png'))
    plt.close()

    # Plot 2: top publishers
    top_pub = news['publisher'].value_counts().head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_pub.values, y=top_pub.index)
    plt.title('Top 20 publishers by article count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'top_publishers.png'))
    plt.close()

    # Plot 3: daily article volume
    df = news.dropna(subset=['date']).copy()
    daily = df.set_index('date').resample('D').size()
    plt.figure(figsize=(12,4))
    daily.plot()
    plt.title('Daily article count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'daily_volume.png'))
    plt.close()

    # Add VADER sentiment and plot distribution
    analyzer = SentimentIntensityAnalyzer()
    news['vader_compound'] = news['headline'].fillna('').apply(lambda t: analyzer.polarity_scores(t)['compound'])
    plt.figure(figsize=(8,4))
    sns.histplot(news['vader_compound'].dropna(), bins=50)
    plt.title('VADER compound sentiment distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'vader_distribution.png'))
    plt.close()

    # Save top TF-IDF terms to CSV
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(news['headline'].fillna(''))
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idx = np.argsort(sums)[-50:][::-1]
    top_terms = [(terms[i], float(sums[i])) for i in top_idx]
    pd.DataFrame(top_terms, columns=['term','score']).to_csv(os.path.join(OUT, 'top_tfidf_terms.csv'), index=False)

    print('EDA complete — outputs written to', OUT)

if __name__ == '__main__':
    main()
