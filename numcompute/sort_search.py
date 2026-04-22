import numpy as np

def sort(arr,axis = -1,descending=False):
    if descending:
        return np.sort(arr,axis=axis,stable=True)[::-1]
    else:
        return np.sort(arr,axis=axis,stable=True)

def multisort(arr,keys,axis=-1):
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")
    return np.lexsort(keys,axis=axis)

def topk(values, k, largest=True, return_indices=False):
    if k <= 0:
        raise ValueError("k must be a positive integer and greater than 0.")
    if k > values.size:
        raise ValueError("k cannot be greater than the number of elements in the input array.")
    
    if largest:
        indices = np.argpartition(values, -k)[-k:]
    else:
        indices = np.argpartition(values, k)[:k]
    
    if return_indices:
        return indices
    else:
        return values[indices]
    

def quickselect(arr, k):

    arr = np.asarray(arr).copy()

    if k < 0 or k >= len(arr):
        raise ValueError("k out of bounds")

    left, right = 0, len(arr) - 1

    while True:
        pivot_index = _partition(arr, left, right)

        if pivot_index == k:
            return arr[pivot_index]
        elif pivot_index < k:
            left = pivot_index + 1
        else:
            right = pivot_index - 1


def _partition(arr, l, r):
    pivot = arr[r]
    i = l

    for j in range(l, r):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[r] = arr[r], arr[i]
    return i

def binary_search(arr, target):
    
    arr = np.asarray(arr)

    idx = np.searchsorted(arr, target)
    
    found = (
        idx < len(arr)
        and arr[idx] == target
    )

    return int(idx), bool(found)   