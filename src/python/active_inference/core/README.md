# Core Agent Implementations

The `core` module contains the fundamental active inference agent implementations that form the foundation of the Active Inference Simulation Lab.

## Agent Classes

### ActiveInferenceAgent

The primary active inference agent implementation following the Free Energy Principle.

**Features:**
- Variational inference for belief updating
- Expected free energy minimization for planning
- Flexible generative model specification
- Modular design for extensibility

**Basic Usage:**
```python
from active_inference.core import ActiveInferenceAgent

agent = ActiveInferenceAgent(
    state_dim=4,      # Dimensionality of hidden state space
    obs_dim=8,        # Dimensionality of observation space
    action_dim=2      # Dimensionality of action space
)

# Single inference step
observation = np.array([1.0, 0.5, -0.2, 0.8, 0.1, -0.3, 0.6, 0.9])
action = agent.act(observation)

# Learning update
reward = 1.0
next_observation = np.array([0.8, 0.3, -0.1, 0.9, 0.2, -0.4, 0.7, 1.0])
agent.update(observation, action, reward, next_observation)
```

**Configuration Options:**
```python
agent = ActiveInferenceAgent(
    state_dim=16,
    obs_dim=32,
    action_dim=4,
    # Inference configuration
    inference_method='variational',
    learning_rate=0.01,
    max_iterations=100,
    # Planning configuration
    planning_horizon=5,
    num_trajectories=50,
    # Optional components
    enable_monitoring=True,
    enable_caching=False,
    security_validation=True
)
```

### AdaptiveActiveInferenceAgent

Extended agent with automatic dimensional adaptation and enhanced security validation.

**Key Features:**
- Automatic adaptation to changing observation/action dimensions
- Input validation and sanitization
- Security threat detection
- Error recovery mechanisms
- Performance monitoring

**Usage:**
```python
from active_inference.core import AdaptiveActiveInferenceAgent

agent = AdaptiveActiveInferenceAgent(
    adaptive_dimensions=True,
    security_validation=True,
    agent_id="adaptive_agent_001"
)

# Agent can handle varying input sizes
obs1 = np.random.randn(8)
obs2 = np.random.randn(12)  # Different dimension

action1 = agent.act(obs1)  # Adapts automatically
action2 = agent.act(obs2)  # Adapts to new dimension
```

### PerformanceOptimizedActiveInferenceAgent

High-performance agent with GPU acceleration, intelligent caching, and memory optimization.

**Features:**
- GPU acceleration for matrix operations
- LRU caching for expensive computations
- Memory pooling to reduce allocation overhead
- Batch processing capabilities

**Usage:**
```python
from active_inference.performance import PerformanceOptimizedActiveInferenceAgent

agent = PerformanceOptimizedActiveInferenceAgent(
    state_dim=64,
    obs_dim=128,
    action_dim=16,
    optimization_config={
        'use_gpu': True,
        'enable_caching': True,
        'cache_size': 1000,
        'batch_size': 32,
        'memory_pool_size': 512
    }
)
```

## Belief Representation

### BeliefState

Represents probabilistic beliefs about hidden states.

**Structure:**
```python
@dataclass
class BeliefState:
    mean: np.ndarray          # Expected state values
    covariance: np.ndarray    # Uncertainty representation
    precision: np.ndarray     # Inverse covariance (cached)
    entropy: float           # Information-theoretic measure
    free_energy: float       # Variational free energy
```

**Operations:**
```python
# Create belief state
beliefs = BeliefState(
    mean=np.zeros(4),
    covariance=np.eye(4) * 0.1
)

# Update beliefs
updated_beliefs = inference.update_beliefs(
    prior_beliefs=beliefs,
    observation=observation,
    model=generative_model
)

# Access belief properties
print(f"Mean: {updated_beliefs.mean}")
print(f"Uncertainty: {np.diag(updated_beliefs.covariance)}")
print(f"Free energy: {updated_beliefs.free_energy}")
```

### GenerativeModel

Represents the agent's model of the world generating observations and rewards.

**Components:**
```python
class GenerativeModel:
    def __init__(self, state_dim, obs_dim, action_dim):
        self.prior = PriorDistribution()        # p(s)
        self.likelihood = LikelihoodModel()     # p(o|s)
        self.transition = TransitionModel()     # p(s'|s,a)
        self.preferences = PreferenceModel()    # Goal states
```

**Learning:**
```python
# Model learns from experience
model = agent.generative_model

# Update transition model
model.transition.update(observation, action, next_observation)

# Update likelihood model
model.likelihood.update(next_observation, observation)

# Update preferences
model.preferences.update(observation, reward)
```

## Inference Methods

### Variational Inference

Approximates posterior beliefs using optimization.

**Algorithm:**
```python
from active_inference.inference import VariationalInference

inference = VariationalInference(
    learning_rate=0.01,
    max_iterations=100,
    convergence_threshold=1e-6
)

posterior_beliefs = inference.update_beliefs(
    prior_beliefs, observation, model
)
```

### Particle Filtering

Samples-based inference for complex distributions.

**Algorithm:**
```python
from active_inference.inference import ParticleFilter

inference = ParticleFilter(num_particles=1000)

posterior_beliefs = inference.update_beliefs(
    prior_beliefs, observation, model
)
```

### Kalman Filtering

Optimal filtering for linear Gaussian systems.

**Algorithm:**
```python
from active_inference.inference import KalmanFilter

inference = KalmanFilter()

# Assumes linear Gaussian generative model
posterior_beliefs = inference.update_beliefs(
    prior_beliefs, observation, model
)
```

## Planning and Decision Making

### Expected Free Energy Planning

Plans actions by minimizing expected free energy.

**Algorithm:**
```python
from active_inference.planning import ExpectedFreeEnergyPlanner

planner = ExpectedFreeEnergyPlanner(
    horizon=5,
    num_trajectories=100
)

action = planner.plan_action(beliefs, model)
```

### Sampling-Based Planning

Uses Monte Carlo methods for planning.

**Algorithms:**
```python
from active_inference.planning import SamplingPlanner

# Monte Carlo Tree Search
mcts_planner = SamplingPlanner(
    num_samples=1000,
    selection_method='mcts'
)

# Random sampling
random_planner = SamplingPlanner(
    num_samples=500,
    selection_method='random'
)

action = mcts_planner.plan_action(beliefs, model)
```

### Hierarchical Planning

Multi-timescale planning for complex tasks.

**Algorithm:**
```python
from active_inference.planning import HierarchicalPlanner

planner = HierarchicalPlanner(
    levels=['low', 'mid', 'high'],
    timescales=[1, 5, 15]
)

action = planner.plan_action(beliefs, model)
```

## Health Monitoring

All agents provide comprehensive health monitoring:

```python
# Get agent health status
health = agent.get_health_status()

print(f"Inference healthy: {health['inference_healthy']}")
print(f"Planning healthy: {health['planning_healthy']}")
print(f"Memory usage: {health['memory_usage']} MB")
print(f"Error rate: {health['error_rate']:.3f}")
```

## Configuration Examples

### Basic Research Agent

```python
from active_inference.core import ActiveInferenceAgent

research_agent = ActiveInferenceAgent(
    state_dim=8,
    obs_dim=16,
    action_dim=4,
    inference_method='variational',
    planning_horizon=10,
    enable_monitoring=True
)
```

### Production Agent

```python
from active_inference.core import AdaptiveActiveInferenceAgent

production_agent = AdaptiveActiveInferenceAgent(
    adaptive_dimensions=True,
    security_validation=True,
    agent_id="prod_agent_001",
    fault_tolerance=True
)
```

### High-Performance Agent

```python
from active_inference.performance import PerformanceOptimizedActiveInferenceAgent

performance_agent = PerformanceOptimizedActiveInferenceAgent(
    state_dim=128,
    obs_dim=256,
    action_dim=32,
    optimization_config={
        'use_gpu': True,
        'enable_caching': True,
        'batch_size': 64,
        'memory_pool_size': 1024
    }
)
```

## Testing and Validation

### Unit Tests

```python
import pytest
import numpy as np
from active_inference.core import ActiveInferenceAgent

class TestActiveInferenceAgent:
    @pytest.fixture
    def agent(self):
        return ActiveInferenceAgent(state_dim=4, obs_dim=8, action_dim=2)

    def test_initialization(self, agent):
        assert agent.state_dim == 4
        assert agent.obs_dim == 8
        assert agent.action_dim == 2

    def test_action_selection(self, agent):
        observation = np.random.randn(8)
        action = agent.act(observation)

        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))

    def test_learning_update(self, agent):
        obs = np.random.randn(8)
        action = np.random.randn(2)
        reward = 1.0
        next_obs = np.random.randn(8)

        agent.update(obs, action, reward, next_obs)

        # Agent should still function after update
        new_action = agent.act(next_obs)
        assert new_action is not None
```

### Performance Benchmarks

```python
import time
import numpy as np
from active_inference.core import ActiveInferenceAgent

def benchmark_agent_performance():
    agent = ActiveInferenceAgent(state_dim=16, obs_dim=32, action_dim=4)

    # Benchmark inference speed
    observations = [np.random.randn(32) for _ in range(1000)]

    start_time = time.time()
    for obs in observations:
        action = agent.act(obs)
    end_time = time.time()

    inference_time = end_time - start_time
    throughput = len(observations) / inference_time

    print(f"Inference throughput: {throughput:.1f} obs/sec")
    print(f"Average latency: {1000 / throughput:.1f} ms")

    return throughput
```

## API Reference

### ActiveInferenceAgent

**Methods:**
- `__init__(state_dim, obs_dim, action_dim, **kwargs)`: Initialize agent
- `act(observation)`: Select action given observation
- `update(observation, action, reward, next_observation)`: Update agent with experience
- `get_beliefs()`: Get current belief state
- `get_health_status()`: Get agent health status

**Attributes:**
- `state_dim`: Hidden state dimensionality
- `obs_dim`: Observation dimensionality
- `action_dim`: Action dimensionality
- `inference`: Inference method instance
- `planning`: Planning method instance
- `generative_model`: Generative model instance

### BeliefState

**Attributes:**
- `mean`: Expected state values (numpy array)
- `covariance`: State uncertainty (numpy array)
- `precision`: Inverse covariance (numpy array)
- `entropy`: Belief entropy (float)
- `free_energy`: Variational free energy (float)

### GenerativeModel

**Components:**
- `prior`: Prior distribution over states
- `likelihood`: Observation likelihood p(o|s)
- `transition`: State transition p(s'|s,a)
- `preferences`: Goal/reward model

## Troubleshooting

### Common Issues

**NaN Values in Actions:**
```python
# Check observation validity
assert np.all(np.isfinite(observation)), "Observation contains NaN/inf values"

# Verify agent health
health = agent.get_health_status()
if not health['inference_healthy']:
    print("Inference component unhealthy - check model parameters")
```

**Poor Performance:**
```python
# Check agent configuration
print(f"State dim: {agent.state_dim}, Obs dim: {agent.obs_dim}")

# Verify learning is occurring
initial_beliefs = agent.get_beliefs()
# ... run some episodes ...
final_beliefs = agent.get_beliefs()
assert not np.allclose(initial_beliefs.mean, final_beliefs.mean), "Beliefs not updating"
```

**Memory Issues:**
```python
# Check memory usage
health = agent.get_health_status()
if health['memory_usage'] > 1000:  # 1GB
    print("High memory usage - consider using PerformanceOptimizedActiveInferenceAgent")
```

## Contributing

When adding new agent implementations:

1. Extend `ActiveInferenceAgent` base class
2. Implement required methods: `act()`, `update()`, `get_beliefs()`
3. Add comprehensive unit tests
4. Update this README with new agent documentation
5. Ensure compatibility with existing interfaces

## References

- [Active Inference: A Process Theory](https://arxiv.org/abs/1504.00789)
- [The Free Energy Principle for Intelligence](https://www.fil.ion.ucl.ac.uk/~karl/The-free-energy-principle.pdf)
- [Variational Inference in Active Inference](https://www.sciencedirect.com/science/article/pii/S0022249621000089)

