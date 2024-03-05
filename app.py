import flask
from flask import Flask, render_template, request
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import plotly
import plotly.express as px
import json
import nltk
'''
    L'analisi del sentiment con VADER è particolarmente efficace per lavorare con testi brevi, come tweet, recensioni, commenti, che possono contenere linguaggio non formale, emoticon, abbreviazioni, ecc
'''
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

finviz_url = 'https://finviz.com/quote.ashx?t='

def get_news(ticker):
    url = finviz_url + ticker
    req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html.parser')
    news_table = html.find(id='news-table')
    return news_table

def parse_news(news_table):
    parsed_news = []
    for x in news_table.findAll('tr'):
        text = x.a.get_text()
        date_scrape = x.td.text.split()
        if len(date_scrape) == 1:
            time = date_scrape[0]
            date = pd.to_datetime('today').strftime('%Y-%m-%d')
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        parsed_news.append([date, time, text])

    columns = ['date', 'time', 'headline']
    parsed_news_df = pd.DataFrame(parsed_news, columns=columns)
    parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
    # Assicurati che 'date' e 'time' non siano più necessari prima di rimuoverli
    parsed_news_df.drop(['date', 'time'], axis=1, inplace=True, errors='ignore')
    return parsed_news_df

def score_news(parsed_news_df):
    vader = SentimentIntensityAnalyzer()
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    parsed_and_scored_news.rename(columns={"compound": "sentiment_score"}, inplace=True)
    return parsed_and_scored_news

def plot_sentiment(parsed_and_scored_news, ticker, freq='H'):
    mean_scores = parsed_and_scored_news['sentiment_score'].resample(freq).mean()
    title = f"{ticker} {freq}ly Sentiment Scores"
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title=title)
    return fig

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment', methods=['POST'])
def sentiment():
    ticker = request.form['ticker'].upper()
    news_table = get_news(ticker)
    parsed_news_df = parse_news(news_table)
    parsed_and_scored_news = score_news(parsed_news_df)
    fig_hourly = plot_sentiment(parsed_and_scored_news, ticker, 'H')
    fig_daily = plot_sentiment(parsed_and_scored_news, ticker, 'D')
    
    graphJSON_hourly = json.dumps(fig_hourly, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_daily = json.dumps(fig_daily, cls=plotly.utils.PlotlyJSONEncoder)
    
    header= f"Sentimento orario e giornaliero di {ticker} Stock"
    description = "I grafici mostrano i punteggi medi del sentiment..."
    return render_template('sentiment.html', graphJSON_hourly=graphJSON_hourly, graphJSON_daily=graphJSON_daily, header=header, table=parsed_and_scored_news.to_html(classes='data'), description=description)

if __name__ == '__main__':
    app.run(debug=True)
