from __future__ import print_function

#from collections import Counter
from optparse import OptionParser
from time import time
import codecs
import logging
import os
import shutil
import sys

print("Load NTLK...")
from nltk.data import load as nltk_data_load
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

import calibreimport
import propernounextractor

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def convert_to_temp_file(raw_text_file, temp_folder):
    TEMP_FILE = os.path.join(temp_folder, "converted.txt")
    if os.path.isfile(TEMP_FILE):
        os.remove(TEMP_FILE)

    raw_text = propernounextractor.load_and_clean_textfile(raw_text_file)
    wnl = WordNetLemmatizer()
    origs = set()
    converted = set()
    result = []
    if True:
        sentence_data = nltk_data_load('tokenizers/punkt/english.pickle')
        last_thousands = 0
        for sentence in sentence_data.tokenize(raw_text):
            tokenized_sentence = word_tokenize(sentence)
            tagged_sentence = pos_tag(tokenized_sentence)
            for token, tag in tagged_sentence:
                sentence_pos = get_wordnet_pos(tag)
                if sentence_pos is None:
                    lemma = token
                else:
                    lemma = wnl.lemmatize(token, pos=sentence_pos)
                new_lemma = not lemma in converted
                origs.add(token)
                converted.add(lemma)
                thousands = len(converted) / 1000
                if token != lemma and new_lemma and thousands > last_thousands:
                    last_thousands = thousands
                    print("%s (%s)-> %s  %d -> %d  (%g)" % (token, tag, lemma, len(origs), len(converted), len(converted) * 100.0 / len(origs) ))
                try:
                    if result:
                        if (not (len(lemma) == 1 and lemma in ".,;:?!") and
                            result[-1] != "``" and
                            lemma != "''"):
                            result.append(" ")
                    result.append(str(lemma))
                except UnicodeEncodeError as uee:
                    pass
#                    print(uee)
            result.append("\n")
    else:
        for token in word_tokenize(raw_text):
            lemma = wnl.lemmatize(token)
            origs.add(token)
            converted.add(lemma)
            if token != lemma:
                print("%s -> %s  %d -> %d  (%g%%)" % (token, lemma, len(origs), len(converted), len(converted) * 100.0 / len(origs) ))
            if result and lemma != "." and lemma != "," and result[-1] != "``" and lemma != "''" and lemma != "?" and lemma != "!" and lemma != "...":
                result.append(" ")
            result.append(str(lemma))

    with open(TEMP_FILE, "w") as f:
        f.write("".join(result))

    return TEMP_FILE

DATA_FOLDER = corpus.DATA_FOLDER

def main():

    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer 
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]    

    corpus3_data = calibreimport.get_text_meta_file_tuples()
    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    for (text_file, meta_data_file) in corpus3_data:
        print(text_file)
        new_file = os.path.join(DATA_FOLDER, os.path.basename(text_file))
        print(new_file)
        new_meta_data_file = new_file + ".meta.xml"
        if os.path.isfile(new_file) and os.path.isfile(new_meta_data_file):
            print("Already exists")
        else:
            print("To convert")
            temp_file = convert_to_temp_file(text_file, DATA_FOLDER)
            assert temp_file is not None
            if os.path.isfile(new_file):
                os.remove(new_file)
            os.rename(temp_file, new_file)
            shutil.copyfile(meta_data_file, new_meta_data_file)
    
    return
    
if __name__ == "__main__":
    main()
