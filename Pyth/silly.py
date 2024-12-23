""" def is_magic_square(grid, magic_sum):
    for i in range(3):
        row_sum = col_sum = 0
        for j in range(3):
            row_sum += grid[i][j]
            col_sum += grid[j][i]
        if row_sum != magic_sum or col_sum != magic_sum:
            return False

    diag1_sum = diag2_sum = 0
    for i in range(3):
        diag1_sum += grid[i][i]
        diag2_sum += grid[i][2 - i]
    return diag1_sum == magic_sum and diag2_sum == magic_sum

def backtrack(grid, row, col, used_numbers, magic_sum):
    if row == 3:
        return is_magic_square(grid, magic_sum)

    if col == 3:
        return backtrack(grid, row + 1, 0, used_numbers, magic_sum)

    for num in range(1, 13):
        if num in used_numbers:
            continue
        grid[row][col] = num
        used_numbers.add(num)

        if backtrack(grid, row, col + 1, used_numbers, magic_sum):
            return True
        used_numbers.remove(num)

    return False

def count_magic_squares():
    count = 0
    for magic_sum in range(12, 25):
        grid = [[0] * 3 for _ in range(3)]
        used_numbers = {1, 2}  # Ensure 1 and 2 are present
        if backtrack(grid, 0, 0, used_numbers, magic_sum):
            count += 1
    return count

print(count_magic_squares()) """

"""def is_valid_magic_square(grid):
    magic_sum = 15
    for i in range(3):
        row_sum = col_sum = 0
        for j in range(3):
            row_sum += grid[i][j]
            col_sum += grid[j][i]
        if row_sum != magic_sum or col_sum != magic_sum:
            return False
    diag1_sum = diag2_sum = 0
    for i in range(3):
        diag1_sum += grid[i][i]
        diag2_sum += grid[i][2-i]
    if diag1_sum != magic_sum or diag2_sum != magic_sum:
        return False
    return True

def generate_magic_squares(grid, row, used_numbers):
    if row == 3:
        if is_valid_magic_square(grid) and len(set(sum(grid, []))) <= 8 and 1 in sum(grid, []) and 2 in sum(grid, []):
            # Check for rotations and reflections here (e.g., using canonical labeling)
            # ...
            print(grid)
        return

    for num in range(1, 13):
        if num not in used_numbers:
            grid[row][0] = num
            used_numbers.add(num)
            generate_magic_squares(grid, row + 1, used_numbers)
            used_numbers.remove(num)

# Initialize the grid and used numbers
grid = [[0] * 3 for _ in range(3)]
used_numbers = set()
generate_magic_squares(grid, 0, used_numbers)
"""

"""
from itertools import permutations

def is_magic_square(grid):
    # Checks if a 3x3 grid is a magic square.
    target_sum = sum(grid[0])
    for row in grid[1:]:
        if sum(row) != target_sum:
            return False
    for col in zip(*grid):
        if sum(col) != target_sum:
            return False
    if sum(grid[i][i] for i in range(3)) != target_sum:
        return False
    if sum(grid[i][2-i] for i in range(3)) != target_sum:
        return False
    return True

def count_magic_squares(max_distinct_values):
    # Counts magic squares with at most max_distinct_values.
    count = 0
    for perm in permutations(range(1, 13), 9):
        grid = [perm[:3], perm[3:6], perm[6:]]
        if len(set(grid[i][j] for i in range(3) for j in range(3))) <= max_distinct_values and (1 in grid[0] and 2 in grid[0]):
            if is_magic_square(grid):
                count += 1
    return count

# Assuming rotations and reflections are considered the same, we can divide the count by 8.
result = count_magic_squares(8) // 8
print(result)"""

"""
import random
import itertools

def is_magic_square(grid):
    # Checks if a 3x3 grid is a magic square.
    target_sum = sum(grid[0])
    for row in grid:
        if sum(row) != target_sum:
            return False
    for col in range(3):
        if sum(grid[i][col] for i in range(3)) != target_sum:
            return False
    if sum(grid[i][i] for i in range(3)) != target_sum:
        return False
    if sum(grid[i][2-i] for i in range(3)) != target_sum:
        return False
    return True

def generate_magic_squares(distinct_numbers):
    # Generates magic squares using a given set of distinct numbers.
    magic_squares = []
    for permutation in itertools.permutations(distinct_numbers):
        grid = [[permutation[i*3+j] for j in range(3)] for i in range(3)]
        if is_magic_square(grid):
            magic_squares.append(grid)
    return magic_squares

def main():
    # Place 1 and 2 randomly
    grid = [[0] * 3 for _ in range(3)]
    row1, col1 = random.randint(0, 2), random.randint(0, 2)
    row2, col2 = random.randint(0, 2), random.randint(0, 2)
    grid[row1][col1] = 1
    grid[row2][col2] = 2

    # Determine the target sum
    target_sum = sum(grid[0]) + sum(grid[1]) + sum(grid[2]) - 1 - 2

    # Generate distinct numbers for the remaining cells
    distinct_numbers = [i for i in range(3, 13) if i != target_sum // 3]

    # Generate magic squares using the distinct numbers
    magic_squares = generate_magic_squares(distinct_numbers)

    # Filter magic squares that satisfy all conditions
    valid_squares = []
    for square in magic_squares:
        if all(num in square[0] + square[1] + square[2] for num in [1, 2]):
            valid_squares.append(square)

    print(len(valid_squares))

if __name__ == "__main__":
    main()
"""

"""from itertools import combinations, permutations, combinations_with_replacement
import numpy as np

def find_magic_squares_with_constraints():
    
    def is_magic_square(grid):
        # Checks if a grid is a magic square.
        rows = [sum(row) for row in grid]
        cols = [sum(col) for col in grid.T]
        diag1 = sum(grid[i, i] for i in range(3))
        diag2 = sum(grid[i, 2 - i] for i in range(3))
        return len(set(rows + cols + [diag1, diag2])) == 1

    all_values = list(range(1, 13))
    unique_solutions = []

    # Iterate over all possible placements for 1 and 2
    for pos1, pos2 in combinations(range(9), 2):
        grid = np.zeros((3, 3), dtype=int)
        grid.flat[pos1] = 1
        grid.flat[pos2] = 2

        remaining_positions = [i for i in range(9) if i not in {pos1, pos2}]

        # Generate all valid combinations for the other cells
        for combo in combinations_with_replacement(all_values, 7):
            if len(set(combo + (1, 2))) <= 8:
                for perm in permutations(combo):
                    temp_grid = grid.copy()
                    for idx, val in zip(remaining_positions, perm):
                        temp_grid.flat[idx] = val

                    if is_magic_square(temp_grid):
                        # Check if this solution is unique (consider rotations/reflections)
                        if not any(
                            np.array_equal(temp_grid, np.rot90(sol, k)) or
                            np.array_equal(temp_grid, np.flipud(sol)) or
                            np.array_equal(temp_grid, np.fliplr(sol))
                            for sol in unique_solutions
                        ):
                            unique_solutions.append(temp_grid)

    return len(unique_solutions), unique_solutions

# Find solutions
num_solutions, solutions = find_magic_squares_with_constraints()
print("Number of unique magic squares:", num_solutions)
for solution in solutions:
    print(solution, "\n")"""

from itertools import combinations, permutations, combinations_with_replacement
import numpy as np

def find_magic_squares_with_constraints():
    """
    Finds all magic squares under the constraints:
    1. Both 1 and 2 must appear in the grid.
    2. The grid must contain at most 8 distinct values.
    3. Sums of rows, columns, and diagonals must be equal.
    4. Rotations and reflections are considered equivalent.
    """
    def is_magic_square(grid):
        """Checks if a grid is a magic square."""
        rows = [sum(row) for row in grid]
        cols = [sum(col) for col in grid.T]
        diag1 = sum(grid[i, i] for i in range(3))
        diag2 = sum(grid[i, 2 - i] for i in range(3))
        return len(set(rows + cols + [diag1, diag2])) == 1

    all_values = list(range(1, 13))
    unique_solutions = []

    # Iterate over all possible placements for 1 and 2
    for pos1, pos2 in combinations(range(9), 2):
        grid = np.zeros((3, 3), dtype=int)
        grid.flat[pos1] = 1
        grid.flat[pos2] = 2

        remaining_positions = [i for i in range(9) if i not in {pos1, pos2}]

        # Generate all valid combinations for the other cells
        for combo in combinations_with_replacement(all_values, 7):
            if len(set(combo + (1, 2))) <= 8:
                for perm in permutations(combo):
                    temp_grid = grid.copy()
                    for idx, val in zip(remaining_positions, perm):
                        temp_grid.flat[idx] = val

                    if is_magic_square(temp_grid):
                        # Check if this solution is unique (consider rotations/reflections)
                        if not any(
                            np.array_equal(temp_grid, np.rot90(sol, k)) or
                            np.array_equal(temp_grid, np.flipud(sol)) or
                            np.array_equal(temp_grid, np.fliplr(sol))
                            for sol in unique_solutions
                        ):
                            unique_solutions.append(temp_grid)

    return len(unique_solutions), unique_solutions

# Run the function
num_solutions, solutions = find_magic_squares_with_constraints()

print(f"Number of unique magic squares: {num_solutions}")
for solution in solutions:
    print(solution)
