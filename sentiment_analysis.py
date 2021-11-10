import nltk
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS


# Get words from list
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all


# Extracting word features
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features


# Draw wordcloud
def wordcloud_draw(data, file_name, color='black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                             and not word.startswith('@')
                             and not word.startswith('#')
                             and word != 'RT'])

    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=2500,
                          height=2000
                          ).generate(cleaned_word)

    plt.figure(1, figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(file_name)


def sentiment_analysis():
    tweets = []
    stopwords_set = set(stopwords.words("english"))

    data = pd.read_csv("Dataset/Sentiment.csv")
    data = data[['text', 'sentiment']]

    train, test = train_test_split(data, test_size=0.1)

    train = train[train.sentiment != "Neutral"]

    for index, row in train.iterrows():
        words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
        words_cleaned = [word for word in words_filtered
                         if 'http' not in word
                         and not word.startswith('@')
                         and not word.startswith('#')
                         and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        tweets.append((words_without_stopwords, row.sentiment))

    train_pos = train[train['sentiment'] == 'Positive']
    train_pos = train_pos['text']

    train_neg = train[train['sentiment'] == 'Negative']
    train_neg = train_neg['text']

    w_features = get_word_features(get_words_in_tweets(tweets))

    # Draw wordcloud for features
    wordcloud_draw(w_features,  'features.png', 'white')

    # Draw wordcloud for positive tweet words
    wordcloud_draw(train_pos, 'positive.png', 'white')

    # Draw wordcloud for nagative tweet words
    wordcloud_draw(train_neg, 'negative.png', 'black')


if __name__ == '__main__':
    sentiment_analysis()
