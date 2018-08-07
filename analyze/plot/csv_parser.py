import csv

def parse_csv(filename):
    obj = {}
    spamReader = csv.reader(open(filename))
    header = next(spamReader, None)
    for i in header:
        obj[i] = []
    for row in spamReader:
        for i in range(0, len(row)):
            obj[header[i]].append(float(row[i]))
    return obj
