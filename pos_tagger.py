#!/usr/bin/python
import pprint

RAW_DATA = []
DATA_FILENAME = "id-ud-dev.conllu"

def load_raw_data(filename):
    global RAW_DATA

    with open(filename, "r") as fin:
        for line in fin:
            if (line[0] != '#'):
                line = line.strip('\n')
                row_data = line.split('\t')
                if len(row_data) != 1:
                    RAW_DATA.append(row_data)

def get_features(data, idx):
    return {
        'word': data[idx][1],
        'prev_word': '_' if int(data[idx][0]) == 1 else data[idx - 1][1],
        'next_word': '_' if (idx == len(data) - 1) or (int(data[idx + 1][0]) == 1) else data[idx + 1][1],
        'prev_tag': '_' if int(data[idx][0]) == 1 else data[idx - 1][3],
        'pos_tag': data[idx][3]
    }

if __name__ == '__main__':
    load_raw_data(DATA_FILENAME)
    # pprint.pprint(RAW_DATA)
    for i in range(0, len(RAW_DATA)):
        print get_features(RAW_DATA, i)
