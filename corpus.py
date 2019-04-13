from __future__ import print_function

import os


DATA_FOLDER = "converted_data"

def get_text_meta_file_tuples():
    result = []
    files = os.listdir(DATA_FOLDER)
    file_set = set(files)
    for file in files:
        if file.endswith(".txt"):
            if file + ".meta.xml" in file_set:
                result.append((os.path.join(DATA_FOLDER, file),
                               os.path.join(DATA_FOLDER, file + ".meta.xml")))
    return result

def main():
    assert False
    
if __name__ == "__main__":
    main()
