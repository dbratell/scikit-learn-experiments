from __future__ import print_function

#from collections import Counter
from optparse import OptionParser
from time import time
import codecs
import logging
import os
import shutil
import sys
import json

import calibreimport
import corpus

import requests


DATA_FOLDER = corpus.DATA_FOLDER

ISBNDB_HOST = "isbndb.com"
ISBNDB_ACCESS_KEY = "473NH0M8"

def main():
    corpus3_data = calibreimport.get_text_meta_file_tuples()
    for (text_file, meta_data_file) in corpus3_data:
        isbn = calibreimport.read_isbn(meta_data_file)
#        print(text_file)
#        print(isbn)
        if isbn:
            genres = get_genres(isbn, only_local_data=True)
            print("%s (%s) -> %s" % (isbn, text_file, str(genres)))
        else:
            print("No isbn for " + text_file)
            
def get_genres(isbn, verbose=False, only_local_data=False):
    CACHE_FILE = "get_genre_cache.json"
    if os.path.isfile(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache_data = json.load(f)
    else:
        cache_data = { "requests": {}}

    dirty_cache = False
    url = "http://%s/api/v2/json/%s/book/%s" % (ISBNDB_HOST, ISBNDB_ACCESS_KEY, isbn)
    if url in cache_data["requests"] and not ("error" in cache_data["requests"][url] and "limit exceeded" in cache_data["requests"][url]["error"]):
        if verbose:
            print("Cache hit")
        response_json = cache_data["requests"][url]
#        print(response_json)
    else:
        if only_local_data:
            return None
#            raise Exception("Nothing local")
        try:
            if verbose:
                print("Doing request")
            response = requests.get(url)
            if verbose:
                try:
                    print(response.text)
                except UnicodeEncodeError as uee:
                    print("response.text:")
                    print(uee)
            response_json = response.json()
            cache_data["requests"][url] = response_json
            dirty_cache = True

            # http://isbndb.com/api/books.xml?access_key=12345678&index1=isbn&value1=0596002068
        finally:
            if dirty_cache:
#                print("Writing cache...")
                with open(CACHE_FILE + ".tmp", "w") as f:
                    json.dump(cache_data, f, indent=2)
                if os.path.isfile(CACHE_FILE + ".bak"):
                    os.remove(CACHE_FILE + ".bak")
                if os.path.isfile(CACHE_FILE):
                    os.rename(CACHE_FILE, CACHE_FILE + ".bak")
                os.rename(CACHE_FILE + ".tmp", CACHE_FILE)
#                print("Wrote cache")

#    print(response_json)
    if "error" in response_json:
#        print(response_json["error"])
        return None
    genres = response_json["data"][0]["subject_ids"]
#    print("%s -> %s" % (isbn, str(genres)))
    return genres

if __name__ == "__main__":
    main()
