"""
Callback for monitoring battery exploration during training.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class BatteryExplorationCallback(BaseCallback):
    """
    Callback that monitors battery usage during training and provides logging.
    
    This callback tracks battery power and State of Charge (SoC) during training
    and prints periodic statistics to help monitor battery exploration behavior.
    
    Usage:
        - Automatically logs battery power and SoC every 1000 timesteps
        - Helps verify that the agent is learning to use the battery effectively
        - Provides insights into battery charging/discharging patterns
    """
    
    def __init__(self, verbose: int = 0, battery_reward_scale: float = 0.1):
        """
        Initialize the battery exploration callback.
        
        Args:
            verbose: Verbosity level (0 = silent, 1 = print stats)
            battery_reward_scale: Scale factor for exploration rewards (currently unused)
        """
        super().__init__(verbose)
        self.battery_reward_scale = battery_reward_scale
        self.battery_usage_history = []
        self.soc_history = []
        
    def _on_step(self) -> bool:
        """Called at each training step to monitor battery usage."""
        # Get battery information from the environment
        if hasattr(self.training_env, 'envs'):
            # For vectorized environments
            env = self.training_env.envs[0]
        else:
            env = self.training_env
            
        # Navigate through wrappers to get the base environment
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
            
        if hasattr(base_env, 'simulator'):
            # Get battery power and SOC from simulator state
            full_state = base_env.simulator.state
            dev_p = list(full_state["dev_p"]["MW"].values())
            des_soc = list(full_state["des_soc"]["MWh"].values())
            
            battery_p = dev_p[3]  # Device 3 is the battery
            battery_soc = des_soc[0]  # First (and only) storage device
            
            # Track battery usage history
            self.battery_usage_history.append(battery_p)
            self.soc_history.append(battery_soc)
            
            # Print periodic statistics
            if self.verbose > 0 and self.num_timesteps % 1000 == 0:
                avg_battery_power = np.mean(np.abs(self.battery_usage_history[-100:]))
                avg_soc = np.mean(self.soc_history[-100:])
                print(f"Battery Exploration - Avg Power: {avg_battery_power:.3f} MW, Avg SOC: {avg_soc:.3f} MWh")
                
        return True
