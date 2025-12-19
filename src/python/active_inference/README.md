# Active Inference Python Package

The `active_inference` Python package provides a comprehensive, production-ready implementation of active inference algorithms for building intelligent agents that learn through minimizing free energy.

## Quick Start

```python
from active_inference import ActiveInferenceAgent

# Create an agent
agent = ActiveInferenceAgent(
    state_dim=4,      # Hidden state dimensionality
    obs_dim=8,        # Observation dimensionality
    action_dim=2      # Action dimensionality
)

# Interact with environment
observation = [1.0, 0.5, -0.2, 0.8, 0.1, -0.3, 0.6, 0.9]
action = agent.act(observation)

print(f"Selected action: {action}")
```

## Package Structure

```
src/python/active_inference/
├── core/             # Core agent implementations
├── environments/     # Environment interfaces and wrappers
├── inference/        # Belief updating algorithms
├── planning/         # Action planning and decision making
├── monitoring/       # Telemetry and performance monitoring
├── performance/      # Optimization and acceleration
├── reliability/      # Fault tolerance and error handling
├── scalability/      # Distributed processing and scaling
├── security/         # Input validation and threat detection
├── research/         # Advanced algorithms and experiments
├── utils/            # Utility functions and helpers
├── deployment/       # Production deployment tools
└── cli/              # Command-line interface
```

## Core Components

### Active Inference Agents

- **ActiveInferenceAgent**: Standard active inference implementation
- **AdaptiveActiveInferenceAgent**: Dimensional adaptability with security validation
- **PerformanceOptimizedActiveInferenceAgent**: GPU acceleration and caching

### Environment Integration

- **GridWorld**: Simple grid-based environments for testing
- **GymWrapper**: OpenAI Gym environment compatibility
- **MuJoCoWrapper**: Physics-based simulation support

### Inference Methods

- **VariationalInference**: Gradient-based posterior approximation
- **ParticleFilter**: Sequential Monte Carlo sampling
- **KalmanFilter**: Optimal linear Gaussian filtering

### Planning Algorithms

- **ExpectedFreeEnergyPlanner**: Free energy minimization
- **SamplingPlanner**: Monte Carlo tree search and random sampling
- **HierarchicalPlanner**: Multi-timescale planning

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/your-org/active-inference-sim-lab.git
cd active-inference-sim-lab

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.9+
- NumPy
- SciPy
- PyTorch (optional, for GPU acceleration)
- Gymnasium (for environment integration)

## Usage Examples

### Basic Training Loop

```python
from active_inference import ActiveInferenceAgent
from active_inference.environments import GridWorld
import numpy as np

# Setup
agent = ActiveInferenceAgent(state_dim=8, obs_dim=25, action_dim=4)  # 5x5 grid
env = GridWorld(size=(5, 5), goals=[(4, 4)], obstacles=[(2, 2)])

# Training
for episode in range(100):
    observation = env.reset()
    total_reward = 0

    while True:
        # Agent action
        action = agent.act(observation.flatten())

        # Environment step (convert action to coordinates)
        action_idx = np.argmax(action)
        action_map = [np.array([0,1]), np.array([0,-1]), np.array([-1,0]), np.array([1,0])]
        env_action = action_map[action_idx]

        next_obs, reward, done, info = env.step(env_action)
        total_reward += reward

        # Agent learning
        agent.update(observation.flatten(), action, reward, next_obs.flatten())

        if done:
            break
        observation = next_obs

    print(f"Episode {episode}: Total reward = {total_reward}")
```

### Performance Optimization

```python
from active_inference.performance import PerformanceOptimizedActiveInferenceAgent

# High-performance agent with GPU acceleration
agent = PerformanceOptimizedActiveInferenceAgent(
    state_dim=64,
    obs_dim=128,
    action_dim=16,
    optimization_config={
        'use_gpu': True,
        'enable_caching': True,
        'batch_size': 32
    }
)
```

### Monitoring and Telemetry

```python
from active_inference.monitoring import TelemetryCollector

# Setup monitoring
telemetry = TelemetryCollector()
agent = ActiveInferenceAgent(..., telemetry_collector=telemetry)

# Training with monitoring
for episode in range(episodes):
    observation = env.reset()
    while True:
        action = agent.act(observation)
        next_obs, reward, done, info = env.step(action)
        agent.update(observation, action, reward, next_obs)

        if done:
            break
        observation = next_obs

# View metrics
metrics = telemetry.get_metrics()
print(f"Inference operations: {metrics.get('inference_operations_total', 0)}")
```

## Advanced Features

### Research Algorithms

```python
from active_inference.research import HierarchicalTemporalActiveInference

# Multi-timescale hierarchical agent
agent = HierarchicalTemporalActiveInference(
    n_levels=3,
    temporal_scales=[1, 5, 15],
    state_dims=[4, 8, 16]
)
```

### Fault Tolerance

```python
from active_inference.reliability import FaultTolerantActiveInferenceAgent

# Fault-tolerant agent with automatic recovery
agent = FaultTolerantActiveInferenceAgent(
    state_dim=16,
    obs_dim=32,
    action_dim=4,
    fault_tolerance_config={
        'circuit_breaker_timeout': 30,
        'retry_attempts': 3,
        'fallback_enabled': True
    }
)
```

### Distributed Processing

```python
from active_inference.scalability import DistributedActiveInferenceCluster

# Distributed processing cluster
cluster = DistributedActiveInferenceCluster()
cluster.start_cluster(num_workers=4)

# Submit inference tasks
task_ids = cluster.submit_batch_inference([
    {'agent_id': 'agent_1', 'observation': obs1, 'model_params': params1},
    {'agent_id': 'agent_2', 'observation': obs2, 'model_params': params2}
])

# Collect results
results = cluster.wait_for_batch_completion(task_ids)
```

## Configuration

### Environment Variables

```bash
# Performance settings
ACTIVE_INFERENCE_GPU_ENABLED=true
ACTIVE_INFERENCE_CACHE_SIZE=1000
ACTIVE_INFERENCE_BATCH_SIZE=64

# Monitoring
ACTIVE_INFERENCE_TELEMETRY_ENABLED=true
ACTIVE_INFERENCE_METRICS_PORT=9090

# Security
ACTIVE_INFERENCE_VALIDATION_LEVEL=high
ACTIVE_INFERENCE_ENCRYPTION_ENABLED=true
```

### Programmatic Configuration

```python
from active_inference import configure

# Global configuration
configure({
    'performance': {
        'gpu_acceleration': True,
        'caching': {'enabled': True, 'size': 1000}
    },
    'monitoring': {
        'telemetry': True,
        'metrics_port': 9090
    },
    'security': {
        'validation_level': 'high',
        'encryption': True
    }
})
```

## API Reference

For complete API documentation, see:

- [Core API](core/)
- [Environments API](environments/)
- [Inference API](inference/)
- [Planning API](planning/)
- [Monitoring API](monitoring/)
- [Performance API](performance/)

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/

# All tests
pytest tests/
```

### Code Quality

```bash
# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines and contribution process.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/active-inference-sim-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/active-inference-sim-lab/discussions)

## Citation

If you use this software in your research, please cite:

```bibtex
@software{active_inference_sim_lab,
  title = {Active Inference Simulation Lab},
  author = {Active Inference Team},
  url = {https://github.com/your-org/active-inference-sim-lab},
  year = {2024}
}
```

