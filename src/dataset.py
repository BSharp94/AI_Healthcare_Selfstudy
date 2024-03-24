# Description: This file contains the functions to read the data from the file and split it into sentences.

def read_data_file(fp: str) -> str:
    with open(fp, "r") as f:
        data = f.read()
    return data

def split_data(data: str) -> list:
    lines = data.split('\n')
    sentences = []
    for line in lines:
        sentences += line.split('.')
    return sentences


def load_data():
    data = read_data_file('dataset/archive/train.dat')
    data += read_data_file('dataset/archive/test.dat')
    return split_data(data)
    