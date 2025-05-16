import nltk
"""
1. Import the data : using nltk library
"""
nltk.download('twitter_samples')
"""
2. Tokenizing the data: split the data into small text called tokens based on whitespaces and punctuations 
"""
from nltk.corpus import twitter_samples #print all of the tweets within a dataset as strings
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

nltk.download('punkt')#Tokenizing
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
"""
3. Normalizing the data: 
Normalization in NLP is the process of cleaning and standardizing text data to reduce variations
and make it more consistent for analysis. It typically includes:

-Lowercasing all words
-Removing punctuation, URLs, and special characters
-Eliminating stopwords (like "the", "is", "and")
-Reducing words to their base form using stemming or lemmatization
"""
nltk.download('wordnet') # is a lexical database for the English language that helps the script determine the base word
nltk.download('avereged_perceptron_tagger') # resource to determine the context of a word in a sentence

"""
4. Removing the noise from the data: hashtags, mentions, emojis, URLs...
"""


import re, string
from nltk.stem.wordnet import WordNetLemmatizer
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token in tweet_tokens:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)


        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
# print(remove_noise(tweet_tokens[0], stop_words))
"""
5. Determining word density: the frequency of a word. 
"""
def get_all_word(cleaned_token_list):
    for tokens in cleaned_token_list:
        for token in tokens:
            yield token
all_pos_words = list(get_all_word(positive_cleaned_tokens_list))
all_neg_words = list(get_all_word(negative_cleaned_tokens_list))
# Frequency of positive words->
from nltk import FreqDist
freq_dist_pos = FreqDist(all_pos_words)
freq_dist_neg = FreqDist(all_neg_words)
print(freq_dist_pos.most_common(10))
print(freq_dist_neg.most_common(10))
print("hi")