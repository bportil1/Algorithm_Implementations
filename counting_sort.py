import random

def counting_sort(arr):

    freqs = dict()

    for val in arr:

        if val not in freqs:
            freqs[val] = 1

        else:
            freqs[val] += 1

    arr = []

    for i in range(1,101):
    
        if i in freqs:

            arr.extend([i]*freqs[i])


    return arr

n = 100

arr = [random.randint(1,100) for _ in range(n)]

arr = counting_sort(arr)

print(arr)
