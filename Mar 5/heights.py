class Solution:
    # @param A: list of integers (heights)
    # @param B: list of integers (number of people in front with greater or equal heights)
    # @return a list of integers representing the reconstructed order
    def order(self, A, B):
        arr = []

        # Combine heights and corresponding number of people in front into tuples
        for x, y in zip(A, B):
            arr.append((x, y))

        # Sort the list of tuples based on heights
        arr.sort()

        # Use a helper function to reconstruct the order
        return self._helper(arr)
    
    def _helper(self, arr):
        # Base case: if the list is empty, return an empty list
        if not arr:
            return []

        # Another base case: if there is only one person, return a list with their height
        if len(arr) == 1:
            return [arr[0][0]]

        left, right = [], []        
        mid = len(arr) // 2

        # Divide the list into two parts based on the mid-point
        for height, pos in arr:
            rem_left = mid - len(left)
            if pos < rem_left:
                left.append((height, pos))
            else:
                right.append((height, pos - rem_left))

        # Recursively reconstruct the order for the left and right parts
        res = []
        res.extend(self._helper(left))
        res.extend(self._helper(right))
        return res

# Example usage
heights = [5, 3, 2, 6, 1, 4]
in_fronts = [0, 1, 2, 0, 3, 2]

# Create an instance of the Solution class
s = Solution()

# Get the reconstructed order
result = s.order(heights, in_fronts)

# Print the result
print(result)
