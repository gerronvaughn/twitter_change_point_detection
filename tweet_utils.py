import csv
import datetime
import json
import string
import random
from nltk.corpus import stopwords
import math
import statistics
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#VARIABLE DECLARATIONS
tweets_json = 'tweets.json'
stop_words = list(stopwords.words('english'))
start = datetime.datetime(2014,10,1)
#time_steps = 20 


#The data set can be found at https://github.com/fivethirtyeight/russian-troll-tweets

#--------------------CSV-FUNCTIONS-------------------
#These functions were all used initially to convert the data to json.
def create_cleaned_database():
    tweets = import_tweets_from_csv()
    tweets = discard_non_english_tweets(tweets)
    tweets_json = json.dumps(tweets)
    with open('tweets.json', 'w') as json_file:
        json_file.write(tweets_json)

def import_tweets_from_csv():
    tweets =[]
    for csv_file in csv_files:
        tweets.extend(csv_to_dict(csv_file))
    return tweets

def csv_to_dict(f):
    with open(f) as csv_file:
        reader = csv.DictReader(csv_file)
        tweets = []
        for _tweet in reader:
            tweet = dict(_tweet)
            tweets.append(tweet)
    for tweet in tweets:
        process_tweet(tweet)
    return tweets

def discard_non_english_tweets(tweets):
    english_tweets =[]
    for tweet in tweets:
        if tweet['language'] == 'English':
            english_tweets.append(tweet)
    return english_tweets

#--------------------JSON-FUNCTIONS------------------
#These functions are used to pull data from the json file
def import_tweets_from_json():
    with open(tweets_json, 'r') as f:
        tweets_str = f.read()
    tweets = json.loads(tweets_str)
    for tweet in tweets:
        process_tweet(tweet)
    return tweets

def process_tweet(tweet):
    #tweet['external_author_id'] = int(tweet['external_author_id'])
    tweet['publish_date'] = datetime.datetime.strptime(tweet['publish_date'].strip(),"%m/%d/%Y %H:%M")
    tweet['harvested_date'] = datetime.datetime.strptime(tweet['harvested_date'].strip(),"%m/%d/%Y %H:%M")
    tweet['following'] = int(tweet['following'])
    tweet['followers'] = int(tweet['followers'])
    tweet['updates'] = int(tweet['updates'])
    tweet['retweet'] = bool(tweet['retweet'])
    tweet['new_june_2018'] = bool(tweet['new_june_2018'])
    #tweet['alt_external_id'] = int(tweet['alt_external_id']) 
    tweet['tweet_id'] = int(tweet['tweet_id'])

#--------------------TIME-SERIES-FUNCTIONS------------------
def make_time_series(tweets,start_date, n):
    n += 1
    time_series = []
    time = []
    i = 0
    tweets_len = len(tweets)
    tweets = sorted(tweets, key=lambda x: x['publish_date'])
    dates = make_dates(start_date, tweets[-1]['publish_date'], n)
    for date in dates:
        while tweets[i]['publish_date'] < date:            
            time.append(tweets[i])
            i += 1
            if i == tweets_len:
                break
        time_series.append(time)
        time = []
    return time_series[1:]

def make_dates(min_date, max_date, n):
    delta = max_date - min_date
    dur = delta // n
    return [min_date + dur * i for i in range(n)]

def clean_and_split_sentences(tweet_time_series):
    temp = [[tweet['content'].split() for tweet in tweets] for tweets in tweet_time_series]
    return [[[process_word(word) for word in sentence if process_word(word)] for sentence in tweets] for tweets in temp] 

def break_up_sentences(tweet_time_series):
    word_collections = []
    words = []
    for sentences in tweet_time_series:
        for sentence in sentences:
            for word in sentence:
                words.append(word)
        word_collections.append(words)
        words = []
    return word_collections

def process_word(word):
    processed_word = word.lower().strip()
    if '@' in processed_word:
        processed_word = ''
    processed_word = processed_word.translate(str.maketrans('','',string.punctuation))
    if processed_word in stop_words:
        processed_word = ''
    elif 'http' in processed_word:
        processed_word = None
    return processed_word

#--------------------TERM-FREQUENCY------------------------
def make_term_frequency(time_step):
    total_word_count = sum([val[1] for val in time_step])
    return [(word_id, math.log(float(count) / float(total_word_count))) for word_id, count in time_step]

def make_tf_time_series(tweets_time_series, keep_only_common_words=True):
    tweets_time_series = break_up_sentences(tweets_time_series)
    tweets_dict = Dictionary(tweets_time_series)
    bow_time_series = [tweets_dict.doc2bow(tweets) for tweets in tweets_time_series]
    tf_time_series = [make_term_frequency(time_step) for time_step in bow_time_series] 
    tf_time_series = [[(tweets_dict.get(tup[0]),tup[1]) for tup in time_step] for time_step in tf_time_series]
    if keep_only_common_words:
        tweets_dict.filter_extremes(no_below=len(tweets_time_series),no_above=1) 
        tf_time_series = [[tup for tup in time_step if tweets_dict.doc2idx([tup[0]])[0] != -1] for time_step in tf_time_series]
    return tf_time_series

def insert_word_into_time_series(word, time_series):
    temp = [dict(time_step) for time_step in time_series]
    for time_step in temp:
        if word not in time_step:
            time_step[word] = 1
    return [list(time_step.items()) for time_step in temp]

#--------------------DISTRUBUTION-----------------
def make_word_embeddings(tweets_time_series,ep=10):
    sentences = []
    models = []
    for time_step in tweets_time_series:
        sentences.extend(time_step)
    global_model = Word2Vec(sentences)
    global_model.save('global_word_embedding')
    for ts in tweets_time_series:
        model = Word2Vec.load('global_word_embedding')
        model.train(ts, total_examples=len(ts), epochs=ep)
        models.append(model)
    return models

def word_embedding_time_series(words, word_approximations, word_embeddings):
    time_series = {}
    for word, approxs in word_approximations.items():
        time_series[word] = []
        first_approx = approxs[0]
        first_embedding_val = word_embeddings[0].wv[word]
        first_time_step_alignment = np.matmul(first_embedding_val,first_approx)
        for word_embedding, M in zip(word_embeddings,approxs):
            current_time_step_alignment = np.matmul(word_embedding.wv[word],M)
            time_series[word].append(cosine_similarity(current_time_step_alignment, first_time_step_alignment))
    return time_series

def get_word_embedding_alligners(words, word_embeddings):
    terminal_embedding = word_embeddings[-1]
    word_embedding_alligning_matrices =  {}
    reg = LinearRegression()
    for word in words:
        word_embedding_alligning_matrices[word] = []
        for word_embedding in word_embeddings:
            current_vectors = most_similar_vectors(word,word_embedding)
            target_vectors = most_similar_vectors(word,terminal_embedding)
            alligning_matrix = reg.fit(current_vectors,target_vectors)
            word_embedding_alligning_matrices[word].append(alligning_matrix.coef_)
    return word_embedding_alligning_matrices

def most_similar_vectors(word, word_embedding):
    most_similar_words = [w[0] for w in word_embedding.wv.similar_by_word(word)]
    most_similar_vectors = [word_embedding.wv.get_vector(w) for w in most_similar_words]
    return most_similar_vectors

def cosine_similarity(vec1, vec2):
    return 1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def make_tuple_time_series(time_series):
    tuple_time_series = []
    for word in time_series.keys():
        tuple_time_series.append([(word,time_step) for time_step in time_series[word]])
    return [[time_step[i] for time_step in tuple_time_series] for i in range(len(tuple_time_series[0]))]


#--------------------CHANGE_POINT_DETECTION-----------------
def change_point_detection(time_series, word, B, c):
    #Preprocessing
    time_series = z_scores(time_series, word) #calculates the z-score for a word at each time_step
    mean_shift_time_series = mean_shift(time_series)
    #Bootstrapping
    boot_strap_samples = [mean_shift(random.sample(time_series,len(time_series))) for _ in range(B)] #samples from time series
    p_values = []   #estimates the probability of a change occuring in the time series at each point
    s = 0
    for i, mean_shift_i in enumerate(mean_shift_time_series):
        for sample in boot_strap_samples:
            if sample[i] > mean_shift_i:
                s += 1
        p_values.append(float(s) / float(B))
        s = 0
    #Change Point Detection
    C = [i for i, val in list(enumerate(time_series))[:-1] if val >= c] #collections indicies when z-score exceeds some user defined values c
    if C:
        p_value = min([p_values[i] for i in C])
        ECP = p_values.index(p_value)
    else:
        p_value = None
        ECP = None
    return p_value, ECP, p_values
        
def z_scores(time_series, word):
    means = [float(sum([val[1] for val in t])) / float(len(t)) for t in time_series] #calculates the mean across all words at each time step.
    variances = [float(sum( [(val[1] - mean) ** 2 for val in t] )) / float(len(t)) for mean, t in zip(means, time_series)] #calculates the variance at each time step
    time_series_for_word = [l[0] for l in [[val[1] for val in t if word == val[0]]for t in time_series]] #extracts the value associated with word at each time_step
    return [(t - mean) / math.sqrt(var) for mean, var, t in zip(means,variances,time_series_for_word)] 

def mean_shift(time_series): 
    n = len(time_series)
    return [statistics.mean(time_series[i:]) - statistics.mean(time_series[:i]) for i in range(1,n)]

def single_word_time_series(word, words_time_series):
    temp = [dict(time_step) for time_step in words_time_series]
    return [time_step[word] for time_step in temp]

def quadratic_variation(time_series): #Used to calculate the qv of real valued time series
    temp = []
    for i in range(len(time_series) - 1):
        temp.append((time_series[i] - time_series[i+1]) ** 2)
    return sum(temp)

def make_graph(word, time_series, dates,title,tf=True):
    y_vals = single_word_time_series(word,time_series)
    plt.title(title)
    plt.xlabel('Time')
    if tf:
        plt.ylabel('Log Probability')
    else:
        plt.ylabel('Distance')
    plt.plot_date(dates,y_vals,'-b')
    plt.show()

def make_p_graph(p_scores, dates,title):
    y_vals = p_scores
    x_vals = dates
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('p-values')
    plt.plot_date(dates,y_vals,'-b')
    plt.show()

