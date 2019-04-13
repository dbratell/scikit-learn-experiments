from __future__ import print_function

from collections import Counter
import os
import codecs
import HTMLParser

CALIBRE_META_FILE = "metadata.opf"
def get_text_meta_file_tuples():
    txt_files = get_text_files()
    res = []

    for txt_file in txt_files:
        metadata_file = os.path.join(os.path.dirname(txt_file),
                                     CALIBRE_META_FILE)
        res.append((txt_file, metadata_file))
    return res

def read_isbn(metadata_file):
    # <dc:identifier opf:scheme="ISBN">9780006173625</dc:identifier>
    with codecs.open(metadata_file, "r", "utf-8") as f:
        for line in f:
            line = line.strip()
            START_TAG = '<dc:identifier opf:scheme="ISBN">'
            END_TAG = "</dc:identifier>"
            if line.startswith(START_TAG) and line.endswith(END_TAG):
                html_parser = HTMLParser.HTMLParser()
                isbn = line[len(START_TAG):-len(END_TAG)]
                isbn = html_parser.unescape(isbn)
                if isbn:
                    return isbn
    return None
    

def read_tags(metadata_file):
    tags = []
    with codecs.open(metadata_file, "r", "utf-8") as f:
        html_parser = HTMLParser.HTMLParser()
        for line in f:
            # Format <dc:subject>Fantasy</dc:subject>
            line = line.strip()
            START_TAG = "<dc:subject>"
            END_TAG = "</dc:subject>"
            if line.startswith(START_TAG) and line.endswith(END_TAG):
                tag = line[len(START_TAG):-len(END_TAG)]
                tag = html_parser.unescape(tag)
                tags.append(tag)
    return tags

CALIBRE_DATA_DIR = r"C:\Users\Daniel\My Documents\calibre"
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
