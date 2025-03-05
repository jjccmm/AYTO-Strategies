import ayto_game
import ayto_solver
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
from functools import partial


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


def create_plots(df, save_plot):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    linestyles = ['-', ':', '--']
    style = {'random': {'color': colors[0], 'linestyle': linestyles[0]},
             'paper_random': {'color': colors[0], 'linestyle': linestyles[1]},
             'paper_clever': {'color': colors[1], 'linestyle': linestyles[1]},
             'pc_random': {'color': colors[0], 'linestyle': linestyles[2]},
             'pc_max': {'color': colors[1], 'linestyle': linestyles[2]},
             'pc_minmax': {'color': colors[2], 'linestyle': linestyles[2]}
             }
    
    df['solved'] = df['state'] == 'solved'
    runs = df['run'].max() + 1
    fig, ax = plt.subplots(2,4)
    ax = ax.flatten()
    
    for i, solver_name in enumerate(['random', 'paper_random', 'paper_clever', 'pc_random', 'pc_max', 'pc_minmax']):  # df['solver'].unique()
        df_solver = df[df['solver'] == solver_name]

        df_solved = df_solver[df_solver['state'] == 'solved']
        percentage_solved = df_solved['run'].nunique() / runs
        mean_events = df_solved['event_number'].mean()
        mean_price_pool = df_solved['price_pool'].mean()
        print(f'{solver_name} solved {percentage_solved:.2%} of the games in {mean_events:.2f} events with a mean price pool of {mean_price_pool:.2f}')
    
        df_box = df_solver[df_solver["event_type"] == 'box']
        df_night = df_solver[df_solver["event_type"] == 'night']
        
        c = style[solver_name]['color']
        ls = style[solver_name]['linestyle']
        
        sns.lineplot(data=df_solver, ax=ax[0], label=f'{solver_name}', color=c, linestyle=ls, x='event_number', y='possibilities')
        ax[0].set_yscale('log')
        ax[0].set_title('Remaining Possibilities')
        
        sns.lineplot(data=df_box, ax=ax[1], label=f'{solver_name}', color=c, linestyle=ls, x='event_number', y='event_percentage_reduction')
        ax[1].set_title('MatchBox: Precentage Reduction')
        ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v *100 :.0f}%"))

        sns.lineplot(data=df_night, ax=ax[2], label=f'{solver_name}', color=c, linestyle=ls, x='event_number', y='event_percentage_reduction')
        ax[2].set_title('MatchNight: Precentage Reduction')
        ax[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v *100 :.0f}%"))

        sns.lineplot(data=df_night, ax=ax[3], label=f'{solver_name}', color=c, linestyle=ls, x='event_number', y='solved', estimator='sum')
        ax[3].set_title('Solved Runs')
        ax[3].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/runs *100 :.0f}%"))

        sns.lineplot(data=df_night, ax=ax[4], label=f'{solver_name}', color=c, linestyle=ls, x='event_number', y='matches')
        ax[4].set_title('Positive MatchBoxes')
        ax[4].set_yticks([0, 1, 2, 3, 4])
        
        sns.lineplot(data=df_night, ax=ax[5], label=f'{solver_name}', color=c, linestyle=ls, x='event_number', y='lights')
        ax[5].set_title('Lights')
        ax[5].set_yticks([0, 2, 4, 6, 8, 10])
        
        sns.lineplot(data=df_night, ax=ax[6], label=f'{solver_name}', color=c, linestyle=ls, x='event_number', y='blackouts')
        ax[6].set_title('Blackouts')
        ax[6].set_yticks([0, 2, 4, 6])

        sns.lineplot(data=df_solver, ax=ax[7], label=f'{solver_name}', color=c, linestyle=ls, x='event_number', y='price_pool')
        ax[7].set_title('Remaining Pricepool')
        ax[7].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k€"))
        
        for a in ax:
            a.set_ylabel('')
            a.set_xticks([0, 5, 10, 15, 20])
            a.set_xticks(np.arange(0, 21, 1), minor=True)
            a.set_xlabel("Event Number")
            a.grid(True, linewidth=0.2)
            a.legend(fontsize=8) 
    
    fig.set_size_inches(18, 8)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.055, top=0.97, wspace=0.24, hspace=0.264),
    if save_plot:
        plt.savefig('ayto_solver_evaluation.png')
    plt.show()


if __name__ == '__main__':
    
    save_to_file = False
    read_from_file = False
    save_plot = False
    
    if read_from_file:
        df = pd.read_csv('ayto_run_summary.csv', header=0)
    else:
        game_logs = []
        num_processes = cpu_count()
        runs = 20
        
        for solver_name, solver in [('random', ayto_solver.random_solver), 
                                    ('paper_random', ayto_solver.random_paper_solver),
                                    ('paper_clever', ayto_solver.clever_paper_solver),
                                    ('pc_random', ayto_solver.random_stateful_solver),
                                    ('pc_max', ayto_solver.max_prob_stateful_solver),
                                    ('pc_minmax', ayto_solver.min_max_prob_stateful_solver)]:
            solver_log = []
            with Pool(num_processes) as pool:
                play_ayto_with_arg = partial(play_ayto, ayto_solver=solver)  
                for log in tqdm(pool.imap_unordered(play_ayto_with_arg, range(runs)), total=runs, desc=f'Running {solver_name}'):
                    solver_log += log
                    
            for entry in solver_log:
                entry['solver'] = solver_name
                
            game_logs += solver_log
            
        df = pd.DataFrame(game_logs)
        
        if save_to_file:
            df.to_csv('ayto_run_summary.csv', index=False)
    
     
    create_plots(df, save_plot)
    



        
    