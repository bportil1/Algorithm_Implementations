import random
import statistics

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

def parallel_arrays_median(arr1: list[int], arr2: list[int], low: int, high: int, mid: int):
    #Find the mid point of the first array
    mid1 = (low + high) //2
    #Find the mid point of the second array
    mid2 = mid - mid1

    #Default values for l1, l1, r1, r2
    l1 = l2 = min(min(arr1), min(arr2))
    r1 = r2 = max(max(arr1), max(arr2))

    #Update l1, l2, l3, l4
    if (mid1 < len(arr1)):
        r1 = arr1[mid1]
    if (mid2 < len(arr2)):
        r2 = arr2[mid2]
    if (mid1 - 1 >= 0):
        l1 = arr1[mid1 - 1]
    if (mid2 - 1 >= 0):
        l2 = arr2[mid2 - 1]

    #Check to see if we've found the median
    if l1 <= r2 and l2 <= r1:
        return (float(max(l1, l2)) + float(min(r1, r2))) / 2

    #If 1f > r2 then we need to remove elements from the right half
    elif l1 > r2:
        median = parallel_arrays_median(arr1, arr2, low, mid1-1, mid)
        
    #If 1f > r2 then we need to remove elements from the left half    
    elif l2 > r1:
        median = parallel_arrays_median(arr1, arr2, mid1+1, high, mid)

    return median

n = 100
min_int = 20224
max_int = 40554
arr = [random.randint(min_int, max_int) for _ in range(n)]

arr = counting_sort(arr, max_int, min_int)

if (len(arr) % 2) == 1:
    arr.append(random.randint(min_int, max_int))

array1 = arr[:int(len(arr)/2)]
array1.sort()
array2 = arr[int(len(arr)/2):]
array2.sort()

midpt = (len(array1) + len(array1) + 1) // 2

low_pt = 0

high_pt = len(array1)

median = parallel_arrays_median(array1, array2, low_pt, high_pt, midpt)

print(statistics.median(arr))

print(median)
