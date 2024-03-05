class Solution:
    def max_adjacent_sum(self, matrix):
        """
        Finds the maximum sum of adjacent elements in a 2D matrix.

        :param matrix: 2D matrix where each row represents a set of elements.
        :return: Maximum sum of adjacent elements.
        """
        if not matrix or len(matrix) == 0:
            return 0

        prev_sum_col_one = 0
        prev_sum_col_two = 0
        current_sum_col_one = 0
        current_sum_col_two = 0

        for j in range(len(matrix[0])):
            if j % 2 == 0:
                # For even columns, update the sums for column one
                prev_sum_col_one, current_sum_col_one = current_sum_col_one, max(prev_sum_col_two, current_sum_col_one) + max(matrix[0][j], matrix[1][j])
            else:
                # For odd columns, update the sums for column two
                prev_sum_col_two, current_sum_col_two = current_sum_col_two, max(prev_sum_col_one, current_sum_col_two) + max(matrix[0][j], matrix[1][j])

        # Return the maximum sum of the two columns
        return max(current_sum_col_one, current_sum_col_two)

# Example usage:
matrix_example = [[1, 2, 3], [4, 5, 6]]
solution_instance = Solution()
result = solution_instance.max_adjacent_sum(matrix_example)
print(result)
