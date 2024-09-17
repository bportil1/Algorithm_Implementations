import random
import statistics
import math
import numpy as np

def counting_sort(arr: list[int], max_int: int , min_int: int) -> list[int]:
    #Dictionary to hold array elements and the number of occurrences
    freqs = dict()

    #Iterate over the given array and count the number of occurrences
    for val in arr:
        if val not in freqs:
            freqs[val] = 1
        else:
            freqs[val] += 1

    #Initialize a new array for the sorted list
    sorted_array = []

    #Iterate over the range [min_int, max_int) and insert the element if it occurred in the original array
    for i in range(min_int, max_int+1):
        if i in freqs:
            sorted_array.extend([i])
            
    return sorted_array

def generate_arrays(array_len: int) -> tuple[list[int], list[int], list[int]]:
    #If array length is odd add one to ensure equal length arrays
    if array_len % 2 == 1:
        array_len += 1

    #Generate random array with distinct values
    joined_array = np.random.permutation(50000)[:array_len]

    #Split and sort arrays
    array1 = joined_array[:len(joined_array)//2]
    array1 = counting_sort(array1, max(array1), min(array1))
    array2 = joined_array[len(joined_array)//2:]
    array2 = counting_sort(array2, max(array2), min(array2))

    joined_array.sort()

    return joined_array, array1, array2

#def parallel_arrays_min_queries(n: int, arr1: int, arr2: int, arr1_ind: int, arr2_ind:  int, query_count) -> int:
def parallel_arrays_min_queries(n: int, arr1: int, arr2: int, arr1_ind: int, arr2_ind:  int, query_count) -> int:
    #Base Case for recursive calls
    if n == 1:
        return max(arr1[arr1_ind], arr2[arr2_ind]), query_count

    #Update middle pointer for arrays
    k = math.ceil(n/2)
        
    #Query both databases and update pointers for recursive search
    if arr1[arr1_ind+k] < arr2[arr2_ind+k]:
        return parallel_arrays_min_queries(k, arr1, arr2, arr1_ind + int(n/2), arr2_ind, query_count + 2)
    else:
        return parallel_arrays_min_queries(k, arr1, arr2, arr1_ind, arr2_ind + int(n/2), query_count + 2)

def test_driver():    
    #Call function to initialize arrays
    joined_array, array1, array2 = generate_arrays(100)

    #Call to search for median
    median, query_count = parallel_arrays_min_queries(len(array1), array1, array2, 0, 0, 0)

    print("Query Minimizing Parallel Arrays Median Finder Evaluation")
    print("Minimum Query Median: ", median)
    print("Min Query Count: ", query_count)
    print("Middle Values from Joined Array: ", joined_array[len(joined_array)//2], " ", joined_array[len(joined_array)//2+1])

if __name__ == '__main__':
    test_driver()
