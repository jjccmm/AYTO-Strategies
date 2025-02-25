import random
import numpy as np
import math


class random_solver:
    # Purely Random Solver with no state or strategy
    # Can make choices that are valid but already known to be wrong
    def __init__(self):
        pass
    
    def generate_matchbox_input(self):
        return (random.randint(0,9), random.randint(0,9))
    
    def process_matchbox_output(self, input, result):
        pass
        
    def generate_matchnight_input(self):
        return random.sample(range(10), 10)
    
    def process_matchnight_output(self, input, result):
        pass


class random_clever_solver:
    # Random Solver that keeps track of the remaining possibilities 
    # Makes a random choice from the remaining possibilities
    def __init__(self):
        self.remaining_options = faster_permutations(10)
    
    
    def generate_matchbox_input(self):
        random_option = random.choice(self.remaining_options)
        random_index = random.randint(0,9)
        return (random_index, random_option[random_index])
    
    
    def process_matchbox_output(self, mb_input, mb_result):
        index, number = mb_input
        if mb_result:
            self.remaining_options = self.remaining_options[self.remaining_options[:, index] == number]
        else:
            self.remaining_options = self.remaining_options[self.remaining_options[:, index] != number]
        
        
    def generate_matchnight_input(self):
        random_option = random.choice(self.remaining_options)   
        return random_option 
    
    
    def process_matchnight_output(self, mn_input, mn_result):
        possible_lights = np.sum(self.remaining_options == mn_input, axis=1)
        correct_count_mask = (possible_lights == mn_result)
        self.remaining_options = self.remaining_options[correct_count_mask]




def faster_permutations(n):
    # From https://stackoverflow.com/questions/64291076/generating-all-permutations-efficiently
    # empty() is fast because it does not initialize the values of the array
    # order='F' uses Fortran ordering, which makes accessing elements in the same column fast
    perms = np.empty((math.factorial(n), n), dtype=np.uint8, order='F')
    perms[0, 0] = 0

    rows_to_copy = 1
    for i in range(1, n):
        perms[:rows_to_copy, i] = i
        for j in range(1, i + 1):
            start_row = rows_to_copy * j
            end_row = rows_to_copy * (j + 1)
            splitter = i - j
            perms[start_row: end_row, splitter] = i
            perms[start_row: end_row, :splitter] = perms[:rows_to_copy, :splitter]  # left side
            perms[start_row: end_row, splitter + 1:i + 1] = perms[:rows_to_copy, splitter:i]  # right side

        rows_to_copy *= i + 1

    return perms