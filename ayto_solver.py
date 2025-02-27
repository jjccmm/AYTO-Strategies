import random
import numpy as np
import math
from abc import ABC, abstractmethod

class solver(ABC):
    @abstractmethod
    def generate_matchbox_input(self):
        pass
    
    @abstractmethod
    def process_matchbox_output(self, input, result):
        pass
        
    @abstractmethod
    def generate_matchnight_input(self):
        pass
    
    @abstractmethod
    def process_matchnight_output(self, input, result):
        pass

class random_solver(solver):
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


class stateful_solver(solver):
    def __init__(self):
        self.remaining_options = self.faster_permutations(10)
    
    @staticmethod
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

    def process_matchbox_output(self, mb_input, mb_result):
        index, number = mb_input
        if mb_result:
            self.remaining_options = self.remaining_options[self.remaining_options[:, index] == number]
        else:
            self.remaining_options = self.remaining_options[self.remaining_options[:, index] != number]
            
    def process_matchnight_output(self, mn_input, mn_result):
        possible_lights = np.sum(self.remaining_options == mn_input, axis=1)
        correct_count_mask = (possible_lights == mn_result)
        self.remaining_options = self.remaining_options[correct_count_mask]        

    def generate_match_probabilites(self):
        count_matrix = np.zeros((10,10), dtype=np.uint32)
        for i in range(10):
            column = self.remaining_options[:,i]
            unique, counts = np.unique(column, return_counts=True)
            count_matrix[unique, i] = counts
        return count_matrix / len(self.remaining_options) 
    
    @abstractmethod
    def generate_matchnight_input(self):
        pass
    
    @abstractmethod
    def generate_matchbox_input(self):
        pass


class random_stateful_solver(stateful_solver):
    # Random Solver that keeps track of the remaining possibilities 
    # Makes a random choice from the remaining possibilities
    def __init__(self):
        super().__init__()
    
    def generate_matchbox_input(self):
        random_option = random.choice(self.remaining_options)
        random_index = random.randint(0,9)
        return (random_index, random_option[random_index])
        
    def generate_matchnight_input(self):
        random_option = random.choice(self.remaining_options)   
        return random_option 
    
    
    
class max_prob_stateful_solver(stateful_solver):
    def __init__(self):
        super().__init__()
        
    def generate_matchbox_input(self):
        match_probs = self.generate_match_probabilites()
        # We dont want to pick a match that is already known to be correct
        match_probs[match_probs == 1] = 0
        number, index = np.unravel_index(np.argmax(match_probs, axis=None), match_probs.shape)
        return (index, number)
    
    
    def generate_matchnight_input(self):
        match_probs = self.generate_match_probabilites()
        # get the highest probability in each column
        max_probs = np.max(match_probs, axis=0)
        # get the index sorted by the highest probability
        sorted_indices = np.argsort(-max_probs)   
        
        options = self.remaining_options.copy()
        for index in sorted_indices:
            index_probs = match_probs[:, index]
            # get the index of highest probability for the current column 
            max_index = np.argmax(index_probs)
            new_options = options[options[:, index] == max_index]
            if len(new_options) > 0:
                options = new_options
            else:
                break
            
        random_option = random.choice(options)   
        return random_option 



class min_max_prob_stateful_solver(stateful_solver):
    # Solvers that tries to minimize the worst case for each event
    def __init__(self):
        super().__init__()    
        self.matching_night_number = 0

    def generate_matchbox_input(self):
        # We want to find matches close to 50% probability and send them to the match box
        # For those ~50:50 matches we have a worst case of 50% probability, which is the best we can do 
        match_probs = self.generate_match_probabilites()
        
        match_probs[match_probs == 0] = 1
        match_probs = np.abs(match_probs - 0.5)
        
        
        number, index = np.unravel_index(np.argmin(match_probs, axis=None), match_probs.shape)
        return (index, number)
    
    
    def generate_matchnight_input(self):
        # For each remaining option we want to check how many options would remain for each possible lights that would be shown
        # We keep track of the worst case for each option and pick the one with the best worst case
        # As this is computationally expensive we only test this for X options. 
        
        self.matching_night_number += 1
        if self.matching_night_number > 3:
            max_options = 5000
        else:
            max_options = 1
            
        if len(self.remaining_options) < max_options:
            options_to_test = self.remaining_options.copy()
        else:  
            random_indices = np.random.choice(self.remaining_options.shape[0], size=max_options, replace=False)
            options_to_test = self.remaining_options[random_indices]
            
        best_option = None
        best_count = len(self.remaining_options)
        
        for option in options_to_test:
            possible_lights = np.sum(self.remaining_options == option, axis=1)
            _, counts = np.unique(possible_lights, return_counts=True)
            max_count = np.max(counts) 
            if max_count <= best_count:
                best_count = max_count
                best_option = option    
        return best_option

class min_prob_solver(stateful_solver):
    def __init__(self):
        super().__init__()
        
    def generate_matchbox_input(self):
        match_probs = self.generate_match_probabilites()
        # We dont want to pick a match that is already known to be correct
        match_probs[match_probs == 0] = 1
        number, index = np.unravel_index(np.argmin(match_probs, axis=None), match_probs.shape)
        return (index, number)
    
    
    def generate_matchnight_input(self):
        match_probs = self.generate_match_probabilites()
        # We dont want to pick a match that is already known to be wrong
        match_probs[match_probs == 0] = 1
        # shuffle an array with 0 to 9
        random_indices = np.random.permutation(10)
        
        options = self.remaining_options.copy()
        for index in random_indices:
            index_probs = match_probs[:, index]
            # get the index of highest probability for the current column 
            max_index = np.argmin(index_probs)
            new_options = options[options[:, index] == max_index]
            if len(new_options) > 0:
                options = new_options
            else:
                break
            
        random_option = random.choice(options)   
        return random_option 


class medium_prob_solver(stateful_solver):
    def __init__(self):
        super().__init__()
        
    def generate_matchbox_input(self):
        match_probs = self.generate_match_probabilites()

        # We want to find matches close to 50% probability
        match_probs = np.abs(match_probs - 0.5)
        
        match_probs[match_probs == 0] = 1
        number, index = np.unravel_index(np.argmin(match_probs, axis=None), match_probs.shape)
        return (index, number)
    
    
    def generate_matchnight_input(self):
        match_probs = self.generate_match_probabilites()
        # We want to find matches close to 50% probability
        match_probs = np.abs(match_probs - 0.5)
        # shuffle an array with 0 to 9
        
        min_probs = np.min(match_probs, axis=0)
        # get the index sorted by the highest probability
        sorted_indices = np.argsort(min_probs)   
        
        options = self.remaining_options.copy()
        for index in sorted_indices:
            index_probs = match_probs[:, index]
            min_index = np.argmin(index_probs)
            new_options = options[options[:, index] == min_index]
            if len(new_options) > 0:
                options = new_options
            else:
                break
            
        random_option = random.choice(options)   
        return random_option 


