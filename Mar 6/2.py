def is_possible(chapters, n, m, max_time):
    days = 1
    time_required = 0

    for i in range(m):
        if time_required + chapters[i] <= max_time:
            time_required += chapters[i]
        else:
            days += 1
            time_required = chapters[i]
            if days > n:
                return False

    return True

def minimal_max_time(chapters, n, m):
    left = max(chapters)
    right = sum(chapters)

    while left < right:
        mid = (left + right) // 2

        if is_possible(chapters, n, m, mid):
            right = mid
        else:
            left = mid + 1

    return left

# Input
t = int(input("Enter number of test cases: "))
for _ in range(t):
    n, m = map(int, input().split())
    chapters = list(map(int, input().split()))
    
    # Output
    print(minimal_max_time(chapters, n, m))
