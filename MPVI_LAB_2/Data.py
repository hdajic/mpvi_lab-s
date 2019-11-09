import csv

def GetIntegerDataFromCsvFile(fileName):
    data = []
    with open(fileName) as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ' ')
        for row in spamreader:
            data.append([int(row[0]), int(row[1])])
    #print('Data', data)
    return data


def GetFloatDataFromCsvFile(fileName):
    data = []
    with open(fileName) as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ' ')
        for row in spamreader:
            data.append([float(row[0]), float(row[1])])
    return data
       

