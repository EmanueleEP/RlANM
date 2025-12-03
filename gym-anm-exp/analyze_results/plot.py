"""This script plots training curves of discounted returns."""
import matplotlib.pyplot as plt
import numpy as np
import os
from analyze_results.plt_setup import set_rc_params
set_rc_params()
FIGSIZE = (10, 4)
# Path to folders containing training statistics.
folder = 'results'
# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)
agent_groups = [['sac'], ['ppo']]
# Load training stats.
eval_stats = {a: [] for a in [group[0] for group in agent_groups]}
for agent_group in agent_groups:
    agent = agent_group[0]
    agent_path = os.path.join(folder, agent)
    
    # Check if agent directory exists
    if not os.path.exists(agent_path):
        print(f"Warning: Directory '{agent_path}' not found. Skipping {agent}.")
        continue
    
    for run in os.listdir(agent_path):
        if not run.startswith('run_'):
            continue
        timesteps, results, discounted_results, ep_lengths = [], [], [], []
        for a in agent_group:
            f = os.path.join(folder, a, run, 'evaluations.npz')
            d = np.load(f)
            timesteps.append(d['timesteps'])
            
            # Handle different array shapes
            results_data = d['results']
            if results_data.ndim == 3:
                results.append(results_data[:, 0, 0])
            elif results_data.ndim == 2:
                results.append(results_data[:, 0])
            else:
                results.append(results_data)
            
            discounted_data = d['discounted_results']
            if discounted_data.ndim == 3:
                discounted_results.append(discounted_data[:, 0, 0])
            elif discounted_data.ndim == 2:
                discounted_results.append(discounted_data[:, 0])
            else:
                discounted_results.append(discounted_data)
            
            ep_len_data = d['ep_lengths']
            if ep_len_data.ndim == 2:
                ep_lengths.append(ep_len_data[:, 0])
            else:
                ep_lengths.append(ep_len_data)
                
        # Concatenate over multiple training sessions.
        timesteps = np.hstack(timesteps)
        results = np.hstack(results)
        discounted_results = np.hstack(discounted_results)
        ep_lengths = np.hstack(ep_lengths)
        d = np.array([timesteps, results, discounted_results, ep_lengths])
        eval_stats[agent].append(d)
# Clean statistics.
for agent, d in eval_stats.items():
    if len(d) == 0:
        print(f"Warning: No data found for {agent}. Skipping.")
        continue
    t_min = np.min([x.shape[1] for x in d])
    eval_stats[agent] = np.stack([x[:, :t_min] for x in d])

# Remove agents with no data
eval_stats = {k: v for k, v in eval_stats.items() if len(v) > 0}

# debug
for agent, d in eval_stats.items():
    print(f"{agent}: timesteps range from {d[0, 0, 0]/1e6:.2f}M to {d[0, 0, -1]/1e6:.2f}M")


# Plot discounted returns.
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
for agent, d in eval_stats.items():
    # Compute mean and std.
    mu, std = np.mean(d[:, 2, :], axis=0), np.std(d[:, 2, :], axis=0)
    x = (d[0, 0] / 1e6)
    ax.plot(x, mu, label=agent.upper())
    ax.fill_between(x, mu-std, mu+std, alpha=0.5)
    # Compute best mean
    idx = np.argmax(mu)
    print(f'Best discounted return for {agent}: {mu[idx]} +/- {std[idx]}')
ax.set_xlim([0, None])
ax.set_xlabel('Timestep (million)')
ax.set_ylabel('Discounted return\ncomputed as in Eqn. (7)')
ax.legend(loc='lower right', fontsize='small')
plt.grid(True)
fig.savefig('figures/discounted_return.pdf', bbox_inches='tight')
# Plot total rewards.
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
for agent, d in eval_stats.items():
    # Compute mean and std.
    mu, std = np.mean(d[:, 1, :], axis=0), np.std(d[:, 1, :], axis=0)
    x = (d[0, 0] / 1e6)
    ax.plot(x, mu, label=agent.upper())
    ax.fill_between(x, mu-std, mu+std, alpha=0.5)
    # Compute best mean
    idx = np.argmax(mu)
    print(f'Best total return for {agent}: {mu[idx]} +/- {std[idx]}')
ax.set_xlim([0, None])
ax.set_xlabel('Timestep (million)')
ax.set_ylabel('Non-discounted return (T=2000)')
ax.legend(loc='lower right')
plt.grid(True)
fig.savefig('figures/nondiscounted_return.pdf', bbox_inches='tight')
# plt.show()
if __name__ == '__main__':
    print('Done!')