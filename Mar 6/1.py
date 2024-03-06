def is_possible(arr, n, k, mid):
    cows_placed = 1
    last_position = arr[0]

    for i in range(1, n):
        if arr[i] - last_position >= mid:
            cows_placed += 1
            last_position = arr[i]

    return cows_placed >= k

def max_min_distance(arr, n, k):
    arr.sort()
    left = 0
    right = arr[-1] - arr[0]

    while left < right:
        mid = (left + right + 1) // 2

        if is_possible(arr, n, k, mid):
            left = mid
        else:
            right = mid - 1

    return left

# Input
n, k = map(int, input().split())
arr = list(map(int, input().split()))

# Output
print(max_min_distance(arr, n, k))
