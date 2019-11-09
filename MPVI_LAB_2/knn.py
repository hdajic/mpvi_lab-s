from collections import  Counter
import math
from Data import GetIntegerDataFromCsvFile
from Data import GetFloatDataFromCsvFile

def knn(data, query, k, distance_func, choice_fun):
    neighbour_distances_and_indices = []

    for index, example in enumerate(data):
        distance = distance_func(example[:-1], query)
        neighbour_distances_and_indices.append((distance, index))
    #print(neighbour_distances_and_indices)
    sorted_neighbour_distances_and_indices = sorted(neighbour_distances_and_indices)
    k_nearest_distances_and_indices = sorted_neighbour_distances_and_indices[:k]
    k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]

    #print(k_nearest_labels, query, k)
    return k_nearest_distances_and_indices, choice_fun(k_nearest_labels)

def mean(labels):
    return sum(labels) / len(labels) #vrati labele koje su u domeni kruga

def mode(labels):
    return Counter(labels).most_common(1)[0][0] 

def neka_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)
    
def clasiffication():
    data = GetIntegerDataFromCsvFile('VoliJabukeUSmoothieju.csv')
    query = [[33], [50], [80]]
    K = [1, 3, 5]
    for i in range(0,3):
        for j in range(0, 3):
            print(knn(data, query[i], K[j], neka_distance, mode))

def regresion():
    data = GetFloatDataFromCsvFile('VisinaTezina.csv')
    K = [1, 3, 5]
    query = [[150.4], [173.4], [190.3]]
    for i in range(0,3):
        for j in range(0, 3):
            print(query[i], knn(data, query[i], K[j], neka_distance, mean)[1])
            #knn(data, query[i], K[j], neka_distance, mean)



def main():
    #clasiffication()
    regresion()



if __name__ == '__main__':
    main()