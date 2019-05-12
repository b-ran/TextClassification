# Tutorial Followed: https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/
# New Dataset Source: http://qwone.com/~jason/20Newsgroups/


import string
import re
import nltk
from keras.models import Sequential
from nltk.corpus import stopwords
from collections import Counter
from os import listdir
from keras_preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical

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

    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    tokenizer.fit_on_texts(train_documents)
    modes = ['binary', 'count', 'tfidf', 'freq']

    for mode in modes:
        train = tokenizer.texts_to_matrix(train_documents, mode)
        test = tokenizer.texts_to_matrix(test_documents, mode)
        n_words = test.shape[1]

        model = Sequential()
        model.add(Dense(50, input_shape=(n_words,), activation='relu'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(train, train_labels, epochs=epochs, verbose=verbose)
        loss, acc = model.evaluate(test, test_labels, verbose=verbose)
        print(mode.capitalize(), str(acc * 100))


vocab_min_occurrence = 2
verbose = 2
epochs = 10
num_classes = 3

# vocab = Counter()
# create_vocab_file(["20_newsgroup/med", "20_newsgroup/space", "20_newsgroup/electronics"])

print("\n--------Improved Baseline With NewsGroup Data--------")

vocab = load_vocab_file("vocab-newsgroup.txt")
electronics_class = get_class("20_newsgroup/sci.electronics", "20_newsgroup/sci.electronics.test", 0)
med_class = get_class("20_newsgroup/sci.med", "20_newsgroup/sci.med.test", 1)
space_class = get_class("20_newsgroup/sci.space", "20_newsgroup/sci.space.test", 2)
model([electronics_class, med_class, space_class])

# tokenizer = Tokenizer()


# words = ["RMarkdown with parameters", "How to Render Rmarkdown embedded in Rmarkdown", "YAML header argument in knitr"]
# labels = [1, 1, 0]
# tokenizer.fit_on_texts(words)
# print(tokenizer.texts_to_matrix(words, 'binary'))
# print('')
# print(tokenizer.texts_to_matrix(words, 'count'))
# print('')
# print(tokenizer.texts_to_matrix(words, 'tfidf'))
# print('')
# print(tokenizer.texts_to_matrix(words, 'freq'))
