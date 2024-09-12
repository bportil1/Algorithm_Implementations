import random

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
            sorted_array.extend([i]*freqs[i])

    return sorted_array

n = 100
min_int = 5
max_int = 738
arr = [random.randint(min_int, max_int) for _ in range(n)]
arr = counting_sort(arr, max_int, min_int)
print(arr)

