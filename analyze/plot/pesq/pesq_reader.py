import numpy as np


def read_pesq(in_dir):
    f = open(in_dir, 'r')
    data = f.read()
    split_data = data.split('\n')
    out = []
    for i in split_data[1:]:
        _out = []
        _data = i.split('\t')
        if len(_data) < 6:
            continue
        _out.append(_data[0])
        _out.append(_data[1].strip())
        _out.append(float(_data[2].strip()))
        _out.append(float(_data[3].strip()))
        _out.append(float(_data[4].strip()))
        _out.append(_data[5].strip())
        out.append(_out)
    return out

# return [[...estimation, ideal, mixes] ...]
def get_moslqo(in_dir, num_of_sources = 3):
    data = read_pesq(in_dir)
    size = len(data) // num_of_sources
    counter = 1
    out = []
    for i in range(0, size):
        _arr = []
        for j in range(0, num_of_sources):
            _arr.append(data[i * num_of_sources + j][3])
        out.append(_arr)
    return out
