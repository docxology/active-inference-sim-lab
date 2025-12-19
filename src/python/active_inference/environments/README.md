# Environment Integration

The `environments` module provides standardized interfaces and wrappers for connecting active inference agents with various simulation environments, from simple grid worlds to complex physics simulations.

## Environment Types

### Built-in Environments

#### GridWorld

Simple grid-based environment for testing and benchmarking.

**Features:**
- Configurable grid sizes (2D and 3D)
- Obstacle placement and navigation challenges
- Goal-directed reward structures
- Partial observability options
- Customizable agent capabilities

**Usage:**
```python
from active_inference.environments import GridWorld

# Create a 10x10 grid world
env = GridWorld(
    size=(10, 10),
    obstacles=[(3, 3), (7, 7)],  # Obstacle positions
    goals=[(9, 9)],               # Goal positions
    partial_observability=True,
    observation_radius=2
)

# Reset environment
observation = env.reset()

# Take action (move right)
action = np.array([1, 0])  # [dx, dy]
next_obs, reward, done, info = env.step(action)

print(f"Reward: {reward}, Done: {done}")
```

**Configuration Options:**
```python
env = GridWorld(
    size=(20, 20),                    # Grid dimensions
    obstacles=[(5, 5), (10, 10)],     # Obstacle positions
    goals=[(15, 15), (18, 19)],       # Multiple goals
    partial_observability=True,        # Limited vision
    observation_radius=3,             # Vision radius
    agent_start=(0, 0),              # Starting position
    max_steps=100,                   # Episode length limit
    reward_structure={
        'goal': 1.0,                  # Goal reward
        'obstacle': -1.0,            # Obstacle penalty
        'step': -0.01,               # Step penalty
        'time_limit': -0.5           # Timeout penalty
    }
)
```

#### MockEnvironment

Fast, configurable environment for testing and development.

**Features:**
- Arbitrary observation and action dimensions
- Configurable noise levels and dynamics
- Research-grade capabilities
- Deterministic or stochastic behavior

**Usage:**
```python
from active_inference.environments import MockEnvironment

# Create mock environment
env = MockEnvironment(
    obs_dim=8,
    action_dim=2,
    reward_noise=0.1,
    observation_noise=0.05,
    temporal_dynamics=True
)

# Configure for research
env.configure_research_mode(
    enable_uncertainty=True,
    add_model_mismatch=True,
    temporal_horizon=10
)

observation = env.reset()
action = np.random.randn(2)
next_obs, reward, done, info = env.step(action)
```

### External Environment Wrappers

#### Gymnasium Wrapper

Standard interface to Gymnasium (formerly OpenAI Gym) environments.

**Features:**
- Automatic action space conversion
- Reward shaping for active inference
- Belief-based reward bonuses
- Uncertainty estimation integration

**Usage:**
```python
import gymnasium as gym
from active_inference.environments import GymWrapper

# Wrap any Gymnasium environment
gym_env = gym.make("CartPole-v1")
env = GymWrapper(
    gym_env,
    add_model_uncertainty=True,
    belief_based_reward=True,
    uncertainty_bonus=0.1
)

# Use with active inference agent
from active_inference.core import ActiveInferenceAgent
agent = ActiveInferenceAgent.from_env(env)

observation = env.reset()
action = agent.act(observation)
next_obs, reward, done, info = env.step(action)
```

**Advanced Configuration:**
```python
env = GymWrapper(
    gym_env,
    action_space_conversion='continuous_to_discrete',  # or 'discrete_to_continuous'
    reward_shaping={
        'belief_bonus_weight': 0.1,
        'uncertainty_penalty': -0.05,
        'goal_proximity_bonus': 0.2
    },
    observation_augmentation={
        'add_uncertainty_estimate': True,
        'add_prediction_error': True,
        'add_goal_progress': True
    }
)
```

#### MuJoCo Wrapper

Physics-based simulation with MuJoCo engine.

**Features:**
- Realistic physics simulation
- Proprioceptive and visual sensors
- Contact forces and dynamics
- Real-time control capabilities

**Prerequisites:**
```bash
pip install mujoco
# Install MuJoCo binaries separately
```

**Usage:**
```python
from active_inference.environments import MuJoCoWrapper

# Load humanoid model
env = MuJoCoWrapper(
    model_file="humanoid.xml",
    proprioceptive_noise=0.01,
    visual_occlusion=True,
    frame_skip=5
)

# Multi-modal observations
observation = env.reset()
proprioceptive = observation['proprioceptive']
visual = observation['visual']
```

## Environment Interface

### Standard Protocol

All environments implement a consistent interface:

```python
class ActiveInferenceEnvironment(ABC):
    """Standard interface for active inference environments."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return observation, reward, done, info."""
        pass

    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space specifications."""
        pass

    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """Get action space specifications."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get environment information for active inference."""
        return {}

    def render(self, mode='human'):
        """Render environment (optional)."""
        pass

    def close(self):
        """Clean up environment resources."""
        pass
```

### Active Inference Extensions

Environments can provide additional active inference support:

```python
def get_belief_compatible_obs(self):
    """Get observation formatted for belief updating."""
    raw_obs = self._get_raw_observation()

    return {
        'observation': raw_obs,
        'uncertainty': self._estimate_observation_uncertainty(),
        'prediction_error': self._compute_prediction_error(),
        'goal_progress': self._measure_goal_progress()
    }

def compute_environment_free_energy(self, beliefs, action):
    """Compute environment-specific free energy terms."""
    # Implementation depends on environment
    pass
```

## Environment Configuration

### Research Configurations

Environments can be configured for different research paradigms:

```python
# Uncertainty-driven exploration
env.configure_uncertainty_driven(
    uncertainty_threshold=0.8,
    exploration_bonus=0.2
)

# Goal-directed behavior
env.configure_goal_directed(
    goal_states=[target_state],
    goal_reward=1.0,
    shaping_reward=True
)

# Curiosity-driven learning
env.configure_curiosity_driven(
    novelty_measure="prediction_error",
    curiosity_weight=0.1
)
```

### Performance Optimization

```python
# High-performance configuration
env.configure_performance(
    vectorized_operations=True,
    cache_observations=True,
    parallel_simulation=True
)

# Memory-efficient configuration
env.configure_memory_efficient(
    observation_compression=True,
    sparse_rewards=True,
    episodic_reset=True
)
```

## Testing and Validation

### Environment Testing Suite

```python
from active_inference.environments import EnvironmentTester

tester = EnvironmentTester()

# Comprehensive environment validation
results = tester.test_environment(env, [
    'reset_functionality',
    'step_consistency',
    'reward_stability',
    'observation_space',
    'action_space',
    'active_inference_compatibility'
])

print(f"Environment tests passed: {results['passed']}/{results['total']}")
```

### Benchmarking Tools

```python
from active_inference.environments import EnvironmentBenchmarker

benchmarker = EnvironmentBenchmarker()

# Performance benchmarking
performance = benchmarker.benchmark_performance(env, num_episodes=100)

print(f"Mean episode length: {performance['episode_length']['mean']}")
print(f"Mean reward: {performance['reward']['mean']}")
print(f"Environment FPS: {performance['fps']}")
```

## Custom Environment Development

### Creating Custom Environments

```python
import numpy as np
from active_inference.environments import ActiveInferenceEnvironment

class CustomEnvironment(ActiveInferenceEnvironment):
    """Example custom environment."""

    def __init__(self, config):
        self.config = config
        self.state = None
        self.steps = 0

    def reset(self):
        """Reset environment."""
        self.state = np.zeros(self.config['state_dim'])
        self.steps = 0
        return self._get_observation()

    def step(self, action):
        """Execute action."""
        # Update state based on action
        self.state += action * self.config['action_scale']

        # Add noise
        self.state += np.random.normal(0, self.config['noise_std'], self.state.shape)

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        done = self._is_done()

        # Update step counter
        self.steps += 1

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Get current observation."""
        return self.state.copy()

    def _compute_reward(self):
        """Compute reward based on state."""
        # Example: reward for being close to target
        target = np.ones_like(self.state)
        distance = np.linalg.norm(self.state - target)
        return -distance

    def _is_done(self):
        """Check if episode is done."""
        return self.steps >= self.config['max_steps']

    def get_observation_space(self):
        """Get observation space info."""
        return {
            'shape': (self.config['state_dim'],),
            'dtype': np.float64,
            'range': [-10, 10]
        }

    def get_action_space(self):
        """Get action space info."""
        return {
            'shape': (self.config['state_dim'],),
            'dtype': np.float64,
            'range': [-1, 1]
        }
```

### Environment Registration

```python
from active_inference.environments import EnvironmentRegistry

# Register custom environment
registry = EnvironmentRegistry()
registry.register('custom_env', CustomEnvironment)

# Create from registry
env_config = {'state_dim': 4, 'action_scale': 0.1, 'noise_std': 0.05, 'max_steps': 100}
env = registry.create('custom_env', env_config)
```

## Integration Examples

### Research Pipeline

```python
from active_inference import create_research_setup

# Create complete research environment
agent, env, monitor = create_research_setup(
    environment_type="grid_world",
    agent_config={
        "inference_method": "variational",
        "planning_horizon": 10,
        "enable_monitoring": True
    },
    environment_config={
        "size": (20, 20),
        "partial_observability": True,
        "uncertainty_regions": True
    }
)

# Research loop
for episode in range(1000):
    obs = env.reset()
    episode_reward = 0

    while True:
        # Get belief-compatible observation
        belief_obs = env.get_belief_compatible_obs()

        # Agent inference and action
        action = agent.act(belief_obs)

        # Environment step
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward

        # Research metrics
        metrics = agent.get_research_metrics()
        monitor.record_metrics(metrics)

        if done:
            break

    print(f"Episode {episode}: Reward = {episode_reward}")
```

### Production Deployment

```python
from active_inference import create_production_setup

# Create production environment
agent, env, health_monitor = create_production_setup(
    environment_type="mujoco",
    model_file="robot.xml",
    agent_config={
        "optimization_level": "high",
        "fault_tolerance": True,
        "monitoring_enabled": True
    }
)

# Production loop
while True:
    try:
        obs = env.get_production_obs()
        action = agent.act(obs)
        env.execute_action(action)

        # Health monitoring
        if not health_monitor.is_healthy():
            agent.trigger_recovery()

    except Exception as e:
        logger.error(f"Production error: {e}")
        agent.handle_production_error(e)
```

## Performance Characteristics

| Environment Type | Complexity | FPS | Memory (MB) | Active Inference Compatible |
|------------------|------------|-----|-------------|----------------------------|
| GridWorld | Low | 1000+ | <1 | ✅ Full |
| MockEnvironment | Configurable | 10000+ | <1 | ✅ Full |
| SocialEnvironment | Medium | 500+ | <5 | ✅ Full |
| GymWrapper | Variable | 100-1000 | 10-100 | ✅ Partial |
| MuJoCoWrapper | High | 50-200 | 100-500 | ✅ Partial |

## Troubleshooting

### Common Issues

**Environment Not Resetting Properly:**
```python
# Ensure proper cleanup
env.close()

# Recreate environment
env = GridWorld(...)
observation = env.reset()
```

**Action/Observation Dimension Mismatches:**
```python
# Check environment specifications
obs_space = env.get_observation_space()
action_space = env.get_action_space()

print(f"Observation shape: {obs_space['shape']}")
print(f"Action shape: {action_space['shape']}")

# Ensure agent matches environment
agent = ActiveInferenceAgent(
    obs_dim=obs_space['shape'][0],
    action_dim=action_space['shape'][0]
)
```

**Performance Issues:**
```python
# Profile environment performance
import time

start_time = time.time()
for _ in range(1000):
    action = np.random.randn(action_dim)
    obs, reward, done, info = env.step(action)
end_time = time.time()

fps = 1000 / (end_time - start_time)
print(f"Environment FPS: {fps}")

if fps < 100:
    print("Consider using simpler environment or optimizing configuration")
```

## Contributing

### Adding New Environments

1. **Implement the Interface:**
   ```python
   from active_inference.environments import ActiveInferenceEnvironment

   class NewEnvironment(ActiveInferenceEnvironment):
       def reset(self):
           # Implementation
           pass

       def step(self, action):
           # Implementation
           pass

       def get_observation_space(self):
           # Implementation
           pass

       def get_action_space(self):
           # Implementation
           pass
   ```

2. **Add Comprehensive Tests:**
   ```python
   class TestNewEnvironment:
       def test_interface_compliance(self):
           # Test all required methods
           pass

       def test_performance_characteristics(self):
           # Test performance benchmarks
           pass

       def test_active_inference_compatibility(self):
           # Test with active inference agents
           pass
   ```

3. **Update Documentation:**
   - Add environment description to this README
   - Include usage examples
   - Document configuration options
   - Add performance characteristics

4. **Register Environment:**
   ```python
   from active_inference.environments import EnvironmentRegistry

   registry = EnvironmentRegistry()
   registry.register('new_environment', NewEnvironment)
   ```

## References

- [OpenAI Gym Documentation](https://gymnasium.farama.org/)
- [MuJoCo Physics Simulation](https://mujoco.org/)
- [Active Inference Environment Design](https://arxiv.org/abs/2004.07219)

