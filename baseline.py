# Tutorial Followed: https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/

import string
import re
import nltk
from keras.models import Sequential
from nltk.corpus import stopwords
from collections import Counter
from os import listdir
from keras_preprocessing.text import Tokenizer
from keras.layers import Dense

nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words("english"))


def load_file(filename):
    file = open(filename, "r")
    text = file.read()
    file.close()
    if verbose > 1:
        print("Loaded File:", filename)
    return text


def save_file(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, "w")
    file.write(data)
    print("File Saved:", filename)
    file.close()


def clean_document(doc):
    words = doc.split()
    clean_doc = list()
    for word in words:
        table = str.maketrans({key: None for key in string.punctuation})
        word = word.translate(table)
        word = word.lower()
        if re.match("[a-zA-z]*", word)[0] == word and 1 < len(word) < 16 and word not in stop_words:
            clean_doc.append(word)
    if verbose > 1:
        print("Cleaned Document:", clean_doc)
    return clean_doc


def add_document_to_vocab(filename, vocab):
    doc = load_file(filename)
    tokens = clean_document(doc)
    vocab.update(tokens)


def process_docs_vocab(dir, vocab):
    print("Vocab Processing:", dir, "Documents")
    count = 0
    for filename in listdir(dir):
        path = dir + "/" + filename
        add_document_to_vocab(path, vocab)
        if verbose > 0:
            count += 1
            print(count, "/", len(listdir(dir)))


def create_vocab_file(paths):
    vocab = Counter()
    for path in paths:
        process_docs_vocab(path, vocab)
    tokens = [i for i, j in vocab.items() if j >= vocab_min_occurrence]
    print("Vocab Size:", len(tokens))
    save_file(tokens, "vocab.txt")


def load_vocab_file(filename):
    vocab = load_file(filename)
    return set(vocab.split())


def doc_to_line(filename, vocab):
    doc = load_file(filename)
    tokens = clean_document(doc)
    tokens = [i for i in tokens if i in vocab]
    return " ".join(tokens)


def process_docs(dir, vocab):
    lines = list()
    for filename in listdir(dir):
        path = dir + '/' + filename
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines


def get_labels(data, num):
    return list([num for _ in range(len(data))])


def get_class(train_dir, test_dir, num):
    train = process_docs(train_dir, vocab)
    train_labels = get_labels(train, num)
    test = process_docs(test_dir, vocab)
    test_labels = get_labels(test, num)
    return train, train_labels, test, test_labels


def model(classes):
    tokenizer = Tokenizer()
    train_documents = list()
    train_labels = list()
    test_documents = list()
    test_labels = list()
    for c in classes:
        train_documents += c[0]
        train_labels += c[1]
        test_documents += c[2]
        test_labels += c[3]

    tokenizer.fit_on_texts(train_documents)
    train = tokenizer.texts_to_matrix(train_documents, "tfidf")
    test = tokenizer.texts_to_matrix(test_documents, "tfidf")
    n_words = test.shape[1]

    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train, train_labels, epochs=epochs, verbose=verbose)
    loss, acc = model.evaluate(test, test_labels, verbose=verbose)
    print(str(acc*100))


vocab_min_occurrence = 2
verbose = 2
epochs = 10

# vocab = Counter()
# create_vocab_file(["txt_sentoken/neg", "txt_sentoken/pos"])

print("--------Baseline With News Reviews--------")
vocab = load_vocab_file("vocab-review.txt")
positive_class = get_class("txt_sentoken/pos", "txt_sentoken/pos-test", 0)
negative_class = get_class("txt_sentoken/neg", "txt_sentoken/neg-test", 1)
model([positive_class, negative_class])
