# AYTO-Strategies

This repository is dedicated to comparing different strategies for the RTL+ show "Are You The One". The goal is to evaluate various solvers that attempt to find the correct matches using different approaches. The repository includes implementations of several solvers, each with its own strategy for making decisions based on the information available from the game events.


## AYTO Game

The `AYTOGame` class in `ayto_game.py` simulates the game environment for "Are You The One". It initializes the game with 10 players and generates all permutations of the numbers 1 to 10, selecting a random permutation as the correct matching solution. The class keeps track of the prize pool, played rounds, solved matches, blackouts, and other game states.

### Methods:
- `match_box(index, number)`: Takes a number and an index as input and reveals if this fits the matching solution.
- `match_night(seating)`: Takes a vector as input and returns the number of elements that match the solution (number of lights).
- `game_state()`: Returns the current state of the game (running, solved, game_over) and the game log.


## Solvers
A solver is able to generate an input for the match box and the matching night and can process the results of both events. There are different `solver` classes in `ayto_solver.py`.

### Methods:
- `generate_matchbox_input()`: Generates and returns a pair for the match box.
- `process_matchbox_output(input, result)`: Processes the result of the match box.  
- `generate_matchnight_input()`: Generates a seating vector for the next matching night. 
- `process_matchnight_output(input, result)`: Processes the result of the matching night.  

### Random Solver
The Random Solver does not use any information from the game events and blindly guesses the matches. It does not keep track of any constraints, previous results or previous decisions made.

### Paper Solver
Paper Solver use a strategy that involves making a table of the information from the matching night and a grid that states for each pair if they can be a match or not. This is a strategy that someone could do on paper, but there will be situations where choices for the game events are made that do not fulfill all constraints known from previous events. 

-  **Random Paper Solver:** For the unknown pairs the random paper solver tries to select random options that still seem to be valid based on the given constrains. 

- **Clever Paper Solver:** The Paper Clever Solver is similar to the Paper Random Solver but instead of selecting random valid pairs, it aims to select pairs with the least amount of options left and therefore higher matching probabilities. This approach increases the chances of finding the correct matches by focusing on the most constrained pairs.

### PC Solver
The PC Solver can keep a complete list of all remaining options that fulfill the given constraints from all previous events. As this list is quite long (10!) this is not possible to do manually and can only be done with a computer. 

- **PC Randoom Solver:** The pc random solver always selects random options from the remaining options. The advantage here is that every choice made always fulfills all known constraints.

- **PC Max Probability Solver:** The pc Max Probability Solver selects the options with the highest matching probability, smaller than 100%. It calculates the probabilities for each pair and chooses the one with the highest likelihood of being a match. 

- **PC Min Max Probability Solver:** The PC Min Max Solver selects options for the events that minimize the possible worst-case scenario. It takes the minimum of the maximum remaining options after the event for every possible outcome of the event. This aims to reduce the worst-case number of remaining options after each event, thereby increasing information win from each event.


## Runner Script
The `ayto_runner.py` script is used to run simulations of the game using different solvers and compare their performance. It uses multiprocessing to run multiple simulations in parallel and generates plots to visualize the results. 

#### Functions:
- `play_ayto(run, ayto_solver)`: Simulates a single run of the game using the specified solver. It plays the match box and matching night events, processes the results, and returns the game log.
- `create_plots(df)`: Creates plots to visualize the performance of the solvers. It generates line plots for various metrics such as remaining possibilities, percentage reduction, solved runs, positive match boxes, lights, blackouts, and remaining prize pool.

## Evaluation 



| Solver      | Solved Runs (%) | Mean Events to solve run  | Mean earned price pool |
|--------------|------------------|--------|----------------------------|
| random       | 0.00%            | nan    | nan                        |
| paper_random | 15.79%           | 16.00  | 83,333.33                  |
| paper_clever | 21.05%           | 17.50  | 75,000.00                  |
| pc_random    | 89.47%           | 16.47  | 117,647.06                 |
| pc_max       | 105.26%          | 16.20  | 135,000.00                 |
| pc_minmax    | 105.26%          | 16.50  | 85,000.00                  |


## 🚀 Usage
1. Install requirements
```bash
    pip install -r requirements.txt
```
2. (Optional) Implement own solver classes in `ayto_solver.py` and add them in the `ayto_runner.py` script.

3. Set the desired number of `runs` in `ayto_runner.py` and run the script:
```bash
    python ayto_runner.py
```
