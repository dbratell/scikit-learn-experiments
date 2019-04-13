from __future__ import print_function

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from collections import Counter
import os
from time import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import numpy as np

def main():
    if False:
        vectorizer = CountVectorizer(min_df=1)
        print(vectorizer)
        corpus = (
            'This is the first document.',
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?',
        )
        print(corpus)
        X = vectorizer.fit_transform(corpus)
        print(vectorizer.get_feature_names())

        print(X)
        print(repr(X))
        print(X.toarray())
    

    corpus2 = (r"C:\Users\Daniel\Documents\calibre\Desmond Bagley\Night of Error (16419)\Night of Error - Desmond Bagley.txt",
               r"C:\Users\Daniel\Documents\calibre\David Baldacci\Last Man Standing (2787)\Last Man Standing - David Baldacci.txt",
               r"C:\Users\Daniel\Documents\calibre\Steve Berry\The Third Secret (5065)\The Third Secret - Steve Berry.txt",)

    # v2 = CountVectorizer(input='filename')
    # X = v2.fit_transform(corpus2)
    # print(repr(X))

    # v3 = CountVectorizer(input='filename',
    #                      ngram_range=(1, 2),
    #                      token_pattern=r'\b\w+\b',
    #                      min_df=1,
    #                      )
    # X = v3.fit_transform(corpus2)
    # print(repr(X))
#    from sklearn.feature_extraction.text import TfidfTransformer
#    transformer = TfidfTransformer(smooth_idf=False)
    from sklearn.feature_extraction.text import TfidfVectorizer
    # v4 = TfidfVectorizer(input='filename',
    #                      ngram_range=(1, 2),
    #                      token_pattern=r'\b\w+\b',
    #                      min_df=1,
    #                      )
    # X = v4.fit_transform(corpus2)
    # print(repr(X))

    corpus3 = get_text_files()
    def is_blacklisted(filename):
        if ("A Song of Stone - Iain" in filename or
            "Complicity - Iain" in filename):
            return True  # Not UTF-8
        if ("Morning Star - Pierce Brown" in filename):
            return True  # BOM?
        return False
    corpus3 = [filename for filename in corpus3 if not is_blacklisted(filename)]
    corpus3_names = [os.path.basename(x) for x in corpus3]
    t0 = time()
    v5 = TfidfVectorizer(input='filename',
                         ngram_range=(1, 2),
                         token_pattern=r'\b\w+\b',
                         min_df=2,
                         max_df=0.5,
                         max_features=10000,
                         stop_words='english',
                         )
    X = v5.fit_transform(corpus3)
#    print(repr(X))
#    print(sorted(v5.vocabulary_.keys())[-100:])
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    n_components = 100 # Or None
    if n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()    

    labels = [x.replace(".", "-").split("-")[-2].strip() for x in corpus3_names]
#    print(labels)
    true_k = np.unique(labels).shape[0]
#    true_k = 4
    verbose = False
    if False:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=verbose)

    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()

#    print(km.labels_)
#    print(km.labels_ == 4)
#    print(np.array([0, 1, 2])[[True, False, True]])
    for i in range(true_k):
        cluster_desc = 'Cluster %i: %s' % (i, ', '.join(np.array(corpus3_names)[km.labels_ == i]))
        print(cluster_desc)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    # print("Adjusted Rand-Index: %.3f"
    #       % metrics.adjusted_rand_score(labels, km.labels_))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    # print()


    vectorizer = v5
    if not False:
        print("Top terms per cluster:")

        if n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(" '%s'" % terms[ind], end='')
            print()        


CALIBRE_DATA_DIR = r"C:\Users\Daniel\My Documents\calibre"
CALIBRE_META_FILE = "metadata.opf"
def get_text_files():
    count = 0
    rootDir = CALIBRE_DATA_DIR
    mobi_count = 0
    txt_count = 0
    epub_count = 0
    ext_counter = Counter()
    txt_files = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        if CALIBRE_META_FILE in fileList:
            for f in fileList:
                filename, ext = os.path.splitext(f.lower())
                ext_counter.update([ext])
                if f.lower().endswith(".txt"):
                    count = count + 1
#                    if count <= 40:
#                        print("%d. Found %s" % (count, dirName))
                    txt_files.append(os.path.join(dirName, f))

    print(ext_counter)
    return txt_files
    
if __name__ == "__main__":
    main()
