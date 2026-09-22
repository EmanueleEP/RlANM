# gym-anm-exp

This repository contains the code used to obtain the experimental results presented in
the paper introducing [`gym-anm`](https://github.com/robinhenry/gym-anm):
```
@misc{henry2021gymanm,
      title={Gym-ANM: Reinforcement Learning Environments for Active Network Management Tasks in Electricity Distribution Systems}, 
      author={Robin Henry and Damien Ernst},
      year={2021},
      eprint={2103.07932},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
which can be accessed [here](https://arxiv.org/abs/2103.07932).

## Code Overview
The code is divided into two folders:
- `rl_agents/` contains the code used to train RL agents,
- `mpc_policies/` contains the code used to run the Model Predictive Control-based policies
  (for more information, see the [`gym-anm` documentation](https://gym-anm.readthedocs.io/en/latest/topics/mpc.html)).

### Reinforcement Learning Agents
The folder `rl_agents/` mainly contains helper functions, callbacks, and wrappers that
were used alongside the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) RL library:
* `callbacks.py` contains callback functions useful for:
  * Displaying a progress bar (`ProgressBarCallback`), and
  * Evaluating the agent's current policy (`EvalCallback`). This is a slightly modified version
    of the [original callback](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py#L261)
    that calls `evaluation.py` instead of the original evaluation function.
* `continue_training.py` can be used to continue training an agent using a saved model.
* `evaluation.py` evaluates the agent's current policy. This is a modified version of the 
  [original function](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py)
  which also computes the discounted returns.
* `hyperparameters.py` contains all hyperparameters.
* `train.py` is the main training script.
* `utils.py` contains utility functions to initialize environments, etc.
* `view_progress.py` can be used to display some statistics about the current state of training.
* `wrappers.py` contains wrappers for the environment:
  * `NormalizeActionWrapper` normalizes the action space to lie in `[-1, 1]`.
  * `TimeLimitWrapper` sets a maximum number of steps that can be taken in an environment before needing a reset.

### Model Predictive Control (MPC) Policies
The script `run_mpc.py` can be used to run either the
[constant forecast policy](https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#constant-forecast) or the
[perfect forecast policy](https://gym-anm.readthedocs.io/en/latest/topics/mpc.html#perfect-forecast) for different
planning steps N (optimization horizon) and safety margin hyperparameters &beta;.

## Running The Code

### Installation & Requirements
Use Python `3.12` or later. The pinned dependencies below match the Gymnasium-based API used by the code.
The dependencies can be installed with:
```
$ pip install -r requirements.txt
```

### RL Agents
Before starting training the agents, you may want to modify certain hyperparameters in `rl_agents/hyperparameters.py`.
In particular, you should specify the folder in which you want the results to be stored `BASE_DIR`.

`hyperparameters.py` is the single place for every tunable value, including the
custom environments' reward weights (`W_*`), SoC bounds, constraint cost scales
(`V_TOL`, `P_TOL`, `Q_TOL`) and constraint limits (`V_TOLERANCE` and
`PQ_LIMIT`, expressed in multiples of those scales; with the Lagrangian or BCP
reward, `Q_LIMIT` is the allowed fraction of the reactive capability in use).
All of them are saved in each run's `hyperparameters.txt`; resuming, evaluating
and visualizing a run reapply the saved values, so later edits to
`hyperparameters.py` only affect new runs (and saved runs that predate a
parameter, which fall back to the current value with a warning).

#### Training
You can then start training your agent with:
```
$ python -m rl_agents.train <ALGO> -s <SEED>
```
where `<ALGO>` can be either `SAC` or `PPO` and `<SEED>` is an optional random seed.
Select the complete experiment configuration with `--method`; no matching
changes in `hyperparameters.py` are required:
```
$ python -m rl_agents.train PPO --method original
$ python -m rl_agents.train PPO --method custom
$ python -m rl_agents.train PPO --method lagrangian
$ python -m rl_agents.train PPO --method bcp
```
`original` uses the upstream ANM6Easy reward, `custom` uses the custom fixed
reward, `lagrangian` uses the custom environment with adaptive multipliers, and
`bcp` uses the custom energy objective with fixed normalized penalties. If
`--method` is omitted, `original` is used.

#### External load and generation profiles

Every Gym-ANM-compatible environment can use profiles selected entirely from the
command line. The environment implementation is left unchanged, so `original`,
`custom`, `lagrangian`, `bcp`, and future environments still keep their own
dynamics and reward definitions.

Pass one NPZ or CSV file for training. Evaluation uses the same source with the
`validation` split unless an evaluation source is supplied explicitly:
```
$ python -m rl_agents.train PPO --method original \
    --profile-file 'profiles/{split}.npz'

$ python -m rl_agents.train PPO --method custom \
    --profile-file profiles/train.csv \
    --eval-profile-file profiles/validation.csv
```
`{split}` is replaced with `train` or `validation`. A directory can also be
passed; the loader looks for `profiles_<split>.npz`, `<split>.npz`, and the
equivalent CSV names. Use `--profile-split` and `--eval-profile-split` to choose
different split names.

An NPZ file contains:

- `load_p_mw`: load injections in MW;
- `gen_p_max_mw`: available non-slack generation in MW;
- optional `load_device_ids` and `gen_device_ids` arrays.

The two profile arrays accept `(timesteps, devices)` or
`(scenarios, timesteps, devices)`. Device-major forms are accepted too. Loads
use the Gym-ANM injection convention and must therefore be non-positive;
available generation must be non-negative. If IDs are omitted, columns follow
the device order in the environment topology.

A CSV uses one row per timestep and wide device columns. The optional
`scenario` column groups rows into equal-length scenarios:
```
scenario,step,load:1,load:3,load:5,generation:2,generation:4
weekday,0,-2.1,-1.8,-2.4,0.0,1.2
weekday,1,-2.0,-1.7,-2.3,0.1,1.3
```
Rows within each scenario are read in file order; `step` is descriptive.

In every format, row 0 of a scenario is interpreted as 00:00 and consecutive
rows must be one environment timestep apart (15 minutes for the ANM6 and BCP
environments). When the environment's auxiliary variable is the time of day,
as in ANM6Easy, it is set from the profile position, so the observed time
matches the injected loads and generation. Scenario lengths should be whole
days; otherwise a warning is raised because the time of day jumps when an
episode wraps around the end of a scenario.

For a database, API, or another custom source, expose a Python callable:
```python
def load_profiles(*, env, split, seed):
    rows = query_database(split=split)
    return {
        "load_p_mw": rows.load_profiles,
        "gen_p_max_mw": rows.generation_profiles,
        "load_device_ids": rows.load_device_ids,
        "gen_device_ids": rows.generator_device_ids,
    }
```
Then reference it without editing the training files:
```
$ python -m rl_agents.train PPO --method lagrangian \
    --profile-provider my_project.profiles:load_profiles
```
The callable receives the unwrapped environment, split name, and profile seed.
It may return the mapping shown above or `rl_agents.profiles.ProfileBank`.
Profile settings are saved with the run and automatically reused when training
is resumed, evaluated, or visualized.

Results will be saved in a new directory `<BASE_DIR>/<ENV_ID>/run_i/` where `i` is replaced by an integer
so as to create a new directory.

For reproducible experiments, an exact output directory and lighter evaluation
settings can be supplied from the command line:
```
$ python -m rl_agents.train PPO --seed 1 \
    --run-dir results/experiment_01/lagrangian/seed_1 \
    --method lagrangian \
    --eval-freq 200000 --eval-episodes 3
```
The output directory must not already exist. At completion, the run contains both
the best validation checkpoint (`best_model.zip`, `training_vec_env`, and
`best_lambda_state.npy` for Lagrangian runs) and the last training state
(`final_model.zip`, `final_vec_env`, and `lambda_state.npy`).

#### Five-seed comparison with Slurm
The `train_compare_array.slurm` job array runs the same five seeds on the custom
Lagrangian method and the original ANM6Easy baseline (10 independent jobs). At
most two jobs run concurrently by default. Submit it from this directory with:
```
$ mkdir -p slurm_logs
$ sbatch train_compare_array.slurm
```
Use a new experiment identifier when launching another batch, because existing
run directories are deliberately not reused:
```
$ sbatch --export=ALL,EXPERIMENT_ID=experiment_02 train_compare_array.slurm
```
`TRAIN_STEPS`, `EVAL_FREQ`, and `EVAL_EPISODES` can be overridden in the same
way. Adjust the array concurrency suffix `%2` only after checking actual CPU and
memory usage on the cluster.

#### Inspecting training status
You can get an overview of the training status of your agents by running:
```
$ python -m rl_agents.view_progress
```
which will print some statistics about the results saved in subfolders of `<BASE_DIR>/<ENV_ID>/`.

#### Visualizing the performance of a trained agent
You can watch a trained agent interact with the environment by running:
```
$ python -m analyze_results.visualize <ALGO> -p <PATH> -s <SLEEP> -T <TIMESTEPS>
```
where `<PATH>` is the path to the run folder (in the form `<BASE_DIR>/<ENV_ID>/run_<i>/`), `<SLEEP>` is
the amount of seconds between updates of the rendering (default is 0.5), and `<TIMESTEPS>` is the number
of timesteps to run.

#### Recording a video of your trained agent
You can record videos of your trained agent by running:
```
$ python -m analyze_results.record_screen <PATH> -l <LENGTH> --fps <FPS>
```
where `<PATH>` is the path to where you want to save the recording, `<LENGTH>` is the duration of the 
recording (seconds) and `<FPS>` is the number of frames/seconds to make.

**NOTE:** The above code will simply record your screen. So you need to have the agent already running 
(see previous section). This feature has not been extensively tested, and similar outcomes can easily
be achieved using tools like [QuickTime Player for Mac](https://libguides.rowan.edu/c.php?g=248114&p=4711659).

### MPC Policies
Either MPC-based policies can be run with the following code:
```
$ python -m mpc_policies.run_mpc <ENV_ID> <POLICY> -T <T> -s <SEED> -o <OUTPUT_FILE>
```
where `<POLICY>` can be either `constant` or `perfect`.

The above code will run the policy in the environment `<ENV_ID>` for `<T>` timesteps, repeat
for different safety margins and optimization horizons, and save the final return of each run
in the file `<OUTPUT_FILE>`.

Note that, as specified in [the documentation](https://gym-anm.readthedocs.io/en/latest/topics/mpc.html),
the policy `perfect` will only work in the environment `ANM6Easy-v0`.


## Questions
If you have any questions regarding this implementation, please feel free to contact me at
robin@robinxhenry.com.
