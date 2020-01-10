import nltk

# This is required to download a ton of stuff needed by nltk (see https://www.nltk.org/data.html)
# The data is placed in c:\nltk_data (a UI is displayed to set the directory)
def download_sample_data():
    nltk.download()

# Tokenization is the processing of segmenting text into sentences or words. In the process,
# we throw away punctuation and extra symbols too.
def tokenize(sentence):
    tokens = nltk.word_tokenize(sentence)

    return tokens

# A natural next step after Tokenization is Stop Words Removal. Stop Words Removal has a similar goal
# as Tokenization: get the text data into a format that’s more convenient for processing. In this case,
# stop words removal removes common language prepositions such as “and”, “the”, “a”, and so on in English.
# This way, when we analyze our data, we’ll be able to cut through the noise and focus in on the words
# that have actual real-world meaning.
#
# Stop words removal can be easily done by removing words that are in a pre-defined list. An important thing
# to note is that there is no universal list of stop words. As such, the list is often created from scratch
# and tailored to the application being worked on.
def remove_stop_words(sentence):
    from nltk.corpus import stopwords

    tokens = tokenize(sentence)

    stop_words = stopwords.words('english')
    filtered_tokens = [w for w in tokens if w not in stop_words]

    return filtered_tokens

# Stemming is another technique for cleaning up text data for processing. Stemming is the process of reducing
# words into their root form. The purpose of this is to reduce words which are spelled slightly differently
# due to context but have the same meaning, into the same token for processing. 
def run_stemming_process(sentence):
    from nltk.corpus import stopwords

    tokens = tokenize(sentence)

    snowball_stemmer = nltk.stem.SnowballStemmer('english')
    tokens = [snowball_stemmer.stem(w) for w in tokens]

    return tokens

# Word embeddings is a way of representing words as numbers, in such a way that words with similar meaning
# have a similar representation. Modern-day word embeddings represent individual words as real-valued
# vectors in a predefined vector space.
#
# All word vectors have the same length, just with different values. The distance between two word-vectors
# is representative of how similar the meaning of the two words is. For example, the vectors of the words
# “cook” and “bake” will be fairly close, but the vectors of the words “football” and “bake” will be quite different.
#
# See the article below for an in-depth tutorial
# https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
def run_word_embedding_process():
    return True

# Term Frequency-Inverse Document Frequency, more commonly known as TF-IDF is a weighting factor often used
# in applications such as information retrieval and text mining. TF-IDF uses statistics to measure how
# important a word is to a particular document.
def calculate_term_frequency(text):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text)

    feature_names = vectorizer.get_feature_names()
    dense_vec = vectors.todense()
    dense_list = dense_vec.tolist()
    tfidf_data = pd.DataFrame(dense_list, columns=feature_names)

    return tfidf_data