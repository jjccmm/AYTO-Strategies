import random
import numpy as np
import math


class AYTOGame:
    def __init__(self):
        self.players = 10
        
        self.price_money = 200_000
        self.match_box_price = 25_000
        self.blackout_penalty = 50_000
        self.blackouts = 0
        
        self.max_events = 20
        self.current_event = 0
        
        self.remaining_options = self.faster_permutations(self.players)
        self.solution = random.choice(self.remaining_options)
        
        self.solved_matches = 0
        self.last_lights = 0
        self.state = 'running'
        
        self.log = [{
            "price_pool": self.price_money,
            "blackouts": self.blackouts,
            "event_number": 0,
            "event_type": 'start',
            "possibilities": len(self.remaining_options),
            "event_combination_reduction": 0,
            "event_percentage_reduction": 0,
            "event_entropy_reduction": 0,
            "lights": self.last_lights,
            "matches": self.solved_matches,
            "state": self.state
        }]
        
      
      
    def match_box(self, index, number):
        self.current_event += 1
        match = self.solution[index] == number
        if match:
            self.remaining_options = self.remaining_options[self.remaining_options[:, index] == number]
            self.solved_matches += 1
        else:
            self.remaining_options = self.remaining_options[self.remaining_options[:, index] != number]
        
        if self.current_event >= self.max_events: 
            self.state = 'game_over'   
        
        self.update_log('box')
        return match
    
    
    def match_night(self, seating):
        self.current_event += 1
        lights = np.sum(self.solution == seating)
        possible_lights = np.sum(self.remaining_options == seating, axis=1)
        correct_count_mask = (possible_lights == lights)
        self.remaining_options = self.remaining_options[correct_count_mask]
        self.last_lights = lights       
              
        if lights <= self.solved_matches:
            self.blackouts += 1
            self.price_money = max(0, self.price_money - self.blackout_penalty)
        
        if self.current_event >= self.max_events: 
            self.state = 'game_over'   
            
        if lights == self.players:
            self.state = 'solved' 
            
        self.update_log('night')
        return lights
    
    
    def game_state(self):
        return {'state': self.state, 'log': self.log}
        
    
    def update_log(self, event_type):
        event_combination_reduction = self.log[-1]["possibilities"] - len(self.remaining_options)
        event_percentage_reduction = event_combination_reduction / self.log[-1]["possibilities"]
        event_entropy_reduction = np.log2(self.log[-1]["possibilities"]) - np.log2(len(self.remaining_options))
        
        new_log = {
            "price_pool": self.price_money,
            "blackouts": self.blackouts,
            "event_number": self.current_event,
            "event_type": event_type,
            "possibilities": len(self.remaining_options),
            "event_combination_reduction": event_combination_reduction,
            "event_percentage_reduction": event_percentage_reduction,
            "event_entropy_reduction": event_entropy_reduction,
            "lights": self.last_lights ,
            "matches": self.solved_matches,
            "state": self.state
        }
        self.log.append(new_log)
        
      
    
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