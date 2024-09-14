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
    for i in range(min_int, max_int):
        if i in freqs:
            sorted_array.extend([i])
    return sorted_array

def generate_arrays(array_len: int) -> tuple[list[int], list[int], list[int]]:
    #If array length is odd add one to ensure equal length arrays
    if array_len % 2 == 1:
        array_len += 1

    #Generate random array with distinct values
    joined_array = np.random.permutation(50000)[:101]
    joined_array = counting_sort(joined_array, max(joined_array), min(joined_array))

    #Split and sort arrays
    array1 = joined_array[:int(len(joined_array)/2)]
    array1.sort()
    array2 = joined_array[int(len(joined_array)/2):]
    array2.sort()

    return joined_array, array1, array2

def parallel_arrays_median(arr1: list[int], arr2: list[int], low: int,
                           high: int, mid: int, query_counter: int) -> tuple[int, int]:

    #Find the mid point of the first array
    mid1 = (low + high) //2
    #Find the mid point of the second array
    mid2 = mid - mid1

    #Default values for l1, l1, r1, r2
    l1 = l2 = min(min(arr1), min(arr2))
    r1 = r2 = max(max(arr1), max(arr2))

    #Four queries made total to the first and last of the subarrays gien by the low and high parameters
    query_counter += 4

    #Update l1, l2, l3, l4
    if (mid1 < len(arr1)):
        r1 = arr1[mid1]
    if (mid2 < len(arr2)):
        r2 = arr2[mid2]
    if (mid1 - 1 >= 0):
        l1 = arr1[mid1 - 1]
    if (mid2 - 1 >= 0):
        l2 = arr2[mid2 - 1]

    #Four queries made total to the array splitting points
    query_counter += 4

    #Check to see if we've found the median
    if l1 <= r2 and l2 <= r1:
        return ((float(max(l1, l2)) + float(min(r1, r2))) / 2), (query_counter)

    #If 1f > r2 then we need to remove elements from the right half
    elif l1 > r2:
        median, query_counter = parallel_arrays_median(arr1, arr2, low, mid1-1, mid, query_counter)
        
    #If 1f > r2 then we need to remove elements from the left half    
    elif l2 > r1:
        median, query_counter = parallel_arrays_median(arr1, arr2, mid1+1, high, mid, query_counter)

    return median, query_counter

def parallel_arrays_min_queries(n: int, arr1: int, arr2: int, arr1_ind: int, arr2_ind:  int, query_count) -> int:
    #Base Case for recursive calls
    if n == 1:
        return min(arr1[arr1_ind], arr2[arr2_ind]), query_count

    #Update middle pointer for arrays
    k = math.ceil((n/2))

    #Query both databases and update pointers for recursive search
    if arr1[arr1_ind+k] < arr2[arr2_ind+k]:
        return parallel_arrays_min_queries(k, arr1, arr2, arr1_ind + (n//2), arr2_ind,  query_count + 2)
    else:
        return parallel_arrays_min_queries(k, arr1, arr2, arr1_ind, arr2_ind + (n//2), query_count + 2)
    
def test_driver():    
    #Call function to initialize arrays
    joined_array, array1, array2 = generate_arrays(5000)

    #Find midpoint for the parallel arrays
    midpt = (len(array1) + len(array1) + 1) // 2

    #Initialize pointers for binary search
    low_pt = 0

    high_pt = len(array1)

    #Variable to count total number of queries made
    query_count = 0

    #Initial function call
    median, query_count = parallel_arrays_median(array1, array2, low_pt, high_pt, midpt, query_count)

    print("Base Parallel Arrays Median Finder Evaluation")
    print("Python Median: ", statistics.median(joined_array))
    print("Custom Function Median: ", median)
    print("Query Count: ", query_count)
    print("Length of Array: ", len(array1))
    print("Log of N: ", math.log(len(array1), 2))
    print("Query Count / Number of Queries per Function Call(8): ", query_count/8)

    median, query_count = parallel_arrays_min_queries(len(array1), array1, array2, 0, 0, 0)

    print("Query Minimizing Parallel Arrays Median Finder Evaluation")
    print("Min Query Median: ", median)
    print("Min Query Count: ", query_count)
    print("Middle Values: ", joined_array[len(joined_array)//2], " ", joined_array[(len(joined_array)//2)-1])

if __name__ == '__main__':
    test_driver()
