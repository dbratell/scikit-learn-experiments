from __future__ import print_function


from collections import Counter
from optparse import OptionParser
from time import time
import codecs
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density

import calibreimport
import corpus
import getgenre
import propernounextractor

MODE_SINGLE_GENRE = "SINGLE-GENRE"
MODE_DUAL_GENRE = "DUAL-GENRE"
MODE_AUTHOR = "AUTHOR"
MODE = MODE_SINGLE_GENRE
#MODE = MODE_DUAL_GENRE
#MODE = MODE_AUTHOR
THE_SINGLE_GENRE = "Mystery"

THE_FIRST_GENRE = THE_SINGLE_GENRE
THE_SECOND_GENRE = "Adventure"

def main():
    corpus3_data = corpus.get_text_meta_file_tuples()
#    corpus3_data = calibreimport.get_text_meta_file_tuples()
#    np.random.shuffle(corpus3_data)
#    corpus3_data = corpus3_data[:200]
    corpus3_y = []
    if MODE == MODE_AUTHOR:
        FALLBACK_AUTHOR = "Other author"
        target_names = [
            FALLBACK_AUTHOR,
            "Agatha Christie",
            "Baldacci",
            "King",
#            "Ken Follett",
            "Pratchett",
            "Scott Card",
            "W. E. Johns",
            "Mercedes Lackey",
            "Koontz",
            "Sayers",
            "Wentworth",
            "Connelly",
            "Grisham",
            "Clancy",
            "Ellis Peters",
        ]
    elif MODE == MODE_SINGLE_GENRE:
        target_names = [
            THE_SINGLE_GENRE,
        ]
        target_names.append("Not " + target_names[0])
    elif MODE == MODE_DUAL_GENRE:
        target_names = [
            THE_FIRST_GENRE,
            THE_SECOND_GENRE
            ]
        target_names.append("%s/%s" % (target_names[0], target_names[1]))
        target_names.append("Neither %s nor %s" % (THE_FIRST_GENRE, THE_SECOND_GENRE))
    else:
        assert False, "Unknown mode" + MODE
    target_names = sorted(target_names)
    for (filename, metafile) in corpus3_data:
        if MODE == MODE_AUTHOR:
            author = FALLBACK_AUTHOR
            for i in range(0, len(target_names)):
                if target_names[i] in filename:
                    author = target_names[i]
                    break
            corpus3_y.append(author)
        elif MODE == MODE_SINGLE_GENRE:
#            tags = filter_tags(calibreimport.read_tags(metafile))
            isbn = calibreimport.read_isbn(metafile)
            genres = None
            if isbn is not None:
#                print(metafile)
#                print(isbn)
                genres = getgenre.get_genres(isbn, only_local_data=True)
#                print(genres)

            if genres is None or genres == []:
                corpus3_y.append(None)
            else:
#                print(genres)
                corpus3_y.append(filter_tags(genres, MODE))
        elif MODE == MODE_DUAL_GENRE:
            corpus3_y.append(filter_tags(calibreimport.read_tags(metafile),
                                         MODE))
        elif False:
            genre_val = 0
            if "Scott Card" in filename:
                genre_val = 1
            elif "Baldacci" in filename or "Koontz" in filename:
                genre_val = 2
            elif "Christie" in filename or "Sawyers" in filename:
                genre_val = 3
            corpus3_y.append(genre_val)
        else:
            assert False, "Unknown mode " + MODE

    if MODE == MODE_SINGLE_GENRE:
        # Filter out data for which we don't know the genre
        new_corpus3_data = []
        new_corpus3_y = []
        for (data, y) in zip(corpus3_data, corpus3_y):
            if y is not None:
                new_corpus3_data.append(data)
                new_corpus3_y.append(y)
        corpus3_data = new_corpus3_data
        corpus3_y = new_corpus3_y

        print(len(corpus3_data))
        print(np.shape(corpus3_y))
#        return
        
    target_encoder = LabelEncoder()
    target_encoder.fit(target_names)
    print(target_encoder.classes_)
    assert (list(target_encoder.inverse_transform(range(len(target_names)))) ==
            list(target_names)), "Encoder is not mapping pos A to A."
    print_y = True
    corpus3_y = target_encoder.transform(corpus3_y)
#    corpus3_y = MultiLabelBinarizer().fit_transform(corpus3_y)
    corpus3_y = np.array(corpus3_y)

    corpus3 = [x[0] for x in corpus3_data]
    corpus3_train, corpus3_test, y_train, y_test = train_test_split(corpus3, corpus3_y, test_size=0.3)
    print(y_train)
    print(y_test)
    corpus3_names = [os.path.basename(x) for x in corpus3]

    t0 = time()
    from sklearn.feature_extraction import text
    my_additional_stop_words = propernounextractor.get_proper_nouns()

    stop_words = text.ENGLISH_STOP_WORDS.union(
        [x.lower() for x in my_additional_stop_words])
#    print(stop_words)
    vectorizer = TfidfVectorizer(input='filename',
                         ngram_range=(1, 2),
                         analyzer='word', # 'char', # 'word'
                         token_pattern=r'\b\w+\b',
#                         strip_accents='unicode', # None
                         min_df=2,
                         max_df=0.5,
                         max_features=min(10000, 200*len(corpus3_train)/3),
                         stop_words=stop_words,
                         decode_error='replace',
#                         tokenizer=LemmaTokenizer(),
                         )
    print(vectorizer)
    print("Reading files from disk")
    X_train = vectorizer.fit_transform(corpus3_train) # , y_train?
    X_test = vectorizer.transform(corpus3_test)
#    print(repr(X))
    print(sorted(vectorizer.vocabulary_.keys())[-100:])
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    scaler = StandardScaler(with_mean=False)    # mean=False since sparse
    # Don't cheat - fit only on training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    # apply same transformation to test data
    X_test = scaler.transform(X_test)

    use_hashing = False
    # mapping from integer feature name to original token string
    if use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names()

    select_chi2 = max(25 * len(target_names), min(100, len(corpus3) / 4)) # 100 or none
    if select_chi2:
        print("Extracting %d best features by a chi-squared test" %
              select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
        print()
        print(feature_names)

    if feature_names:
        feature_names = np.asarray(feature_names)

    def trim(s):
        CONSOLE_WIDTH = 135
        """Trim string to fit on terminal (assuming 120-column display)"""
        return s if len(s) <= CONSOLE_WIDTH else s[:(CONSOLE_WIDTH - 3)] + "..."

    def benchmark(clf):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            print_top10 = True
            if print_top10 and feature_names is not None:
                print("top 10 keywords per class:")
                for i, label in enumerate(target_names):
                    if i >= len(clf.coef_):
                        print("%s: Missing data???" % label)
                        continue
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    try:
                        print(trim("%s: \"%r\"" % (label, '" "'.join(feature_names[top10]))))
                    except UnicodeEncodeError as e:
                        print(e)
                print()

        print_report = True
        if print_report:
            print("classification report:")
            print(target_names)
            print(metrics.classification_report(y_test, pred,
                                                target_names=target_names))

        print_cm = True
        if print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time

    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
#            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=10), "Random forest10"),
            (RandomForestClassifier(n_estimators=20), "Random forest20"),
            (RandomForestClassifier(n_estimators=50), "Random forest50"),
            (RandomForestClassifier(n_estimators=100), "Random forest100"),
            (RandomForestClassifier(n_estimators=200), "Random forest200"),
            ):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf))

    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # # Train Liblinear model
        # results.append(benchmark(LinearSVC(loss='squared_hinge',
        #                                    penalty=penalty,
        #                                         dual=False, tol=1e-3)))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty)))

    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty="elasticnet")))

    # Train NearestCentroid without threshold
    print('=' * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid()))
    # Train sparse Naive2 Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01)))
    bernouilli_nb = BernoulliNB(alpha=.01)
    results.append(benchmark(bernouilli_nb))
    results.append(benchmark(BernoulliNB(alpha=.005)))
    results.append(benchmark(BernoulliNB(alpha=.02)))

    # print('=' * 80)
    # print("LinearSVC with L1-based feature selection")
    # # The smaller C, the stronger the regularization.
    # # The more regularization, the more sparsity.
    # results.append(benchmark(Pipeline([
    #   ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
    #   ('classification', LinearSVC())
    # ])))

    print('=' * 80)
    print("Neural network")
    results.append(benchmark(MLPClassifier(solver='lbfgs', alpha=1e-5,
                                           hidden_layer_sizes=(100, 40),
                                           random_state=1)))

    results.append(benchmark(MLPClassifier(solver='lbfgs', alpha=1e-6,
                                           max_iter=200, verbose=True,
                                           hidden_layer_sizes=(100, 40),
                                           random_state=1)))


    results.append(benchmark(MLPClassifier(solver='lbfgs', alpha=1e-7,
                                           max_iter=200, verbose=True,
                                           hidden_layer_sizes=(50,),
                                           random_state=1)))

    if MODE == MODE_AUTHOR:
        # Show output of Agatha Christie and bernouilli_nb
        assert target_names[0] == "Agatha Christie"
        y_pred = bernouilli_nb.predict(X_test)
        index = 0
        for (guess, correct) in zip(y_pred, y_test):
            if guess != correct:
                if guess == 0:
                    print("Thought book %d was Agatha Christie but it was %s" % (index, corpus3_test[index]))
                elif correct == 0:
                    print("Did not recognize %s as an Agatha Christie novel. Thought it was %s" % (corpus3_test[index], target_names[guess]))
            index += 1
    
    # make some plots

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()


SKIPPED_TAGS = set()
def filter_tags(tags, mode=None):
    if mode is None or mode == MODE_SINGLE_GENRE:
        is_mystery = ("mystery_fiction" in tags or
                      "mystery_thrillers_thrillers" in tags or
                      "mystery_thrillers_general" in tags)
#        if THE_SINGLE_GENRE in tags:
        if is_mystery:
            return THE_SINGLE_GENRE
        return "Not " + THE_SINGLE_GENRE
    elif mode == MODE_DUAL_GENRE:
        if THE_FIRST_GENRE in tags:
            if THE_SECOND_GENRE in tags:
                return "%s/%s" % (THE_FIRST_GENRE, THE_SECOND_GENRE)
            return THE_FIRST_GENRE
        elif THE_SECOND_GENRE in tags:
            return THE_SECOND_GENRE
        return "Neither %s nor %s" % (THE_FIRST_GENRE, THE_SECOND_GENRE)
    assert False
    res = []
    for tag in tags:
        if tag in ("Fantasy", "Adventure", "Science Fiction", "Mystery",
                   "Romance", "Thriller", "Detective", "Adult", "Crime"):
            res.append(tag)
        else:
            if tag not in SKIPPED_TAGS:
                print("Skipping '%s'" % tag)
                SKIPPED_TAGS.add(tag)
    return res
    
if __name__ == "__main__":
    main()
