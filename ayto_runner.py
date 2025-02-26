import ayto_game
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import ayto_solver


def play_ayto(run, ayto_solver):
    # Init Game and Solver
    ayto = ayto_game.AYTOGame()
    solver = ayto_solver()
    
    for _ in range(30):
        # Play Matchbox
        mb_input = solver.generate_matchbox_input()
        mb_result = ayto.match_box(mb_input[0], mb_input[1])
        solver.process_matchbox_output(mb_input, mb_result)
        # Play Matchnight
        mn_input = solver.generate_matchnight_input()
        mn_result = ayto.match_night(mn_input)
        solver.process_matchnight_output(mn_input, mn_result)
        # Check Game Ende
        state = ayto.game_state()
        if state['state'] == 'solved' or state['state'] == 'game_over':
            break
        
    # Add run to log
    log = state['log']
    for entry in log:
        entry['run'] = run
    
    return log


if __name__ == '__main__':

    game_logs = []
    num_processes = cpu_count()
    runs = 250
    
    for solver_name, solver in [('random', ayto_solver.random_solver), 
                                ('random_clever', ayto_solver.random_clever_solver),
                                ('max_prob', ayto_solver.max_prob_solver),
                                ('medium_prob', ayto_solver.medium_prob_solver)]:
        solver_log = []
        with Pool(num_processes) as pool:
            play_ayto_with_arg = partial(play_ayto, ayto_solver=solver)  
            for log in tqdm(pool.imap_unordered(play_ayto_with_arg, range(runs)), total=runs, desc=f'Running {solver_name}'):
                solver_log += log
                
        for entry in solver_log:
            entry['solver'] = solver_name
            
        game_logs += solver_log


    df = pd.DataFrame(game_logs)
    #df.to_csv('ayto.csv', index=False)
    #print(df)
    df['solved'] = df['state'] == 'solved'
    
    fig, ax = plt.subplots(2,4)
    ax = ax.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, solver_name in enumerate(df['solver'].unique()):
        df_solver = df[df['solver'] == solver_name]

        df_solved = df_solver[df_solver['state'] == 'solved']
        percentage_solved = df_solved['run'].nunique() / runs
        mean_events = df_solved['event_number'].mean()
        mean_price_pool = df_solved['price_pool'].mean()
        print(f'{solver_name} solved {percentage_solved:.2%} of the games in {mean_events:.2f} events with a mean price pool of {mean_price_pool:.2f}')
        color = colors[i]
    
        df_box = df_solver[df_solver["event_type"] == 'box']
        df_night = df_solver[df_solver["event_type"] == 'night']
        
        sns.lineplot(data=df_solver, ax=ax[0], label=f'{solver_name}', color=color, x='event_number', y='possibilities')
        ax[0].set_yscale('log')
        
        sns.lineplot(data=df_solver, ax=ax[1], label=f'{solver_name}', color=color, x='event_number', y='price_pool')
    
        sns.lineplot(data=df_box, ax=ax[2], label=f'{solver_name}', color=color, x='event_number', y='event_percentage_reduction')
        
        sns.lineplot(data=df_night, ax=ax[3], label=f'{solver_name}', color=color, x='event_number', y='event_percentage_reduction')

        sns.lineplot(data=df_night, ax=ax[4], label=f'{solver_name}', color=color, x='event_number', y='lights')
        
        sns.lineplot(data=df_night, ax=ax[5], label=f'{solver_name}', color=color, x='event_number', y='blackouts')

        sns.lineplot(data=df_night, ax=ax[6], label=f'{solver_name}', color=color, x='event_number', y='matches')
        
        sns.lineplot(data=df_night, ax=ax[7], label=f'{solver_name}', color=color, x='event_number', y='solved', estimator='sum')
        
        
    
    fig.set_size_inches(18, 8)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.074, top=0.97, wspace=0.36, hspace=0.2),
    plt.show()

        
    