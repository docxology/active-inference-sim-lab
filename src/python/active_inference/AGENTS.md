# Python Active Inference Package - AGENTS

## Package Overview

The `active_inference` Python package provides a comprehensive, production-ready implementation of active inference algorithms. Built on the Free Energy Principle, it offers research-grade implementations with enterprise-level reliability, performance, and scalability.

## Package Structure

```
src/python/active_inference/
├── core/           # Core agent implementations
├── environments/   # Environment wrappers and integration
├── inference/      # Belief updating and inference methods
├── planning/       # Action planning and decision making
├── monitoring/     # Telemetry and health monitoring
├── performance/    # Optimization and caching
├── reliability/    # Fault tolerance and error handling
├── scalability/    # Distributed processing and scaling
├── security/       # Input validation and threat detection
├── research/       # Advanced algorithms and experiments
├── utils/          # Utility functions and helpers
├── deployment/     # Production deployment tools
└── __init__.py     # Package initialization
```

## Core Agent Classes

### ActiveInferenceAgent
Primary agent implementation with variational inference.

**Key Features:**
- Variational belief updating
- Expected free energy minimization
- Flexible state and observation dimensions
- Configurable planning horizons

**Usage:**
```python
from active_inference.core import ActiveInferenceAgent

agent = ActiveInferenceAgent(
    state_dim=4,
    obs_dim=8,
    action_dim=2,
    inference_method="variational",
    planning_horizon=5
)

action = agent.act(observation)
```

### AdaptiveActiveInferenceAgent
Extended agent with dimensional adaptability and security.

**Key Features:**
- Automatic dimension adaptation
- Input validation and sanitization
- Security threat detection
- Error recovery mechanisms

**Usage:**
```python
from active_inference.core import AdaptiveActiveInferenceAgent

agent = AdaptiveActiveInferenceAgent(
    adaptive_dimensions=True,
    security_validation=True,
    agent_id="adaptive_agent"
)
```

## Environment Integration

### Supported Environments
- **Gymnasium**: Standard RL environment interface
- **MuJoCo**: Physics-based simulation
- **Custom Environments**: Research-specific implementations

### Environment Wrappers
```python
from active_inference.environments import GymWrapper, MuJoCoWrapper

# Gym environment
env = GymWrapper(gym.make("CartPole-v1"))

# MuJoCo environment
env = MuJoCoWrapper("humanoid.xml")
```

## Performance Optimization

### Optimization Strategies
- **GPU Acceleration**: CUDA/OpenCL support
- **Intelligent Caching**: LRU and domain-specific caches
- **Memory Pooling**: Efficient resource management
- **Batch Processing**: Parallel inference operations

### PerformanceOptimizedActiveInferenceAgent
```python
from active_inference.performance import PerformanceOptimizedActiveInferenceAgent

agent = PerformanceOptimizedActiveInferenceAgent(
    optimization_config={
        "use_gpu": True,
        "enable_caching": True,
        "batch_size": 64
    }
)
```

## Reliability & Monitoring

### Fault Tolerance
- **Circuit Breakers**: Prevent cascade failures
- **Bulkhead Isolation**: Resource protection
- **Retry Mechanisms**: Automatic error recovery
- **Health Monitoring**: Real-time system health

### Monitoring Integration
```python
from active_inference.monitoring import TelemetryCollector

telemetry = TelemetryCollector()
telemetry.start()

# Automatic metrics collection
agent = ActiveInferenceAgent(enable_monitoring=True)
```

## Research Capabilities

### Advanced Algorithms
- **Hierarchical Active Inference**: Multi-temporal processing
- **Causal Active Inference**: Interventional planning
- **Hybrid Symbolic-Connectionist**: Rule-based + neural reasoning
- **Continual Learning**: Task adaptation without forgetting

### Research Framework
```python
from active_inference.research import HierarchicalTemporalActiveInference

agent = HierarchicalTemporalActiveInference(
    n_levels=3,
    temporal_scales=[1, 5, 15]
)
```

## Security Features

### Input Validation
- **Multi-layer validation**: Type, bounds, semantic checks
- **Threat detection**: Anomaly identification
- **Data sanitization**: Safe input processing

### Security Integration
```python
from active_inference.security import AdaptiveThreatDetector

detector = AdaptiveThreatDetector()
threats = detector.detect_threats(client_id, input_data)
```

## Scalability Features

### Distributed Processing
- **Multi-node coordination**: Cluster-based processing
- **Auto-scaling**: Dynamic resource allocation
- **Load balancing**: Efficient workload distribution

### Distributed Framework
```python
from active_inference.scalability import DistributedActiveInferenceCluster

cluster = DistributedActiveInferenceCluster()
cluster.start_cluster(num_workers=4)
```

## Utility Functions

### Unified Interfaces
- **Logging**: Structured logging with telemetry
- **Validation**: Consistent input validation
- **Configuration**: Environment-based configuration

### Utility Usage
```python
from active_inference.utils import get_logger, validate_input

logger = get_logger("agent")
validate_input(data, "observation", expected_shape=(8,))
```

## Testing and Quality Assurance

### Comprehensive Testing
- **Unit Tests**: Individual component validation
- **Integration Tests**: System-level verification
- **Performance Tests**: Benchmarking and profiling
- **Security Tests**: Vulnerability assessment

### Quality Gates
- **Code Coverage**: 85%+ requirement
- **Security Scanning**: Automated vulnerability detection
- **Performance Benchmarks**: Regression prevention

## Package Dependencies

### Core Dependencies
- **numpy**: Numerical computing and array operations
- **scipy**: Scientific computing and optimization
- **torch**: Neural network support (optional)
- **gymnasium**: Reinforcement learning environments

### Optional Dependencies
- **mujoco**: Physics simulation
- **psutil**: System monitoring
- **prometheus_client**: Metrics collection
- **ray**: Distributed computing

## Configuration Management

### Environment Variables
```bash
# Performance settings
USE_GPU=true
BATCH_SIZE=64

# Monitoring
ENABLE_TELEMETRY=true
METRICS_PORT=9090

# Security
SECURITY_LEVEL=high
VALIDATION_STRICT=true
```

### Configuration Classes
```python
from active_inference import OptimizationConfig, SecurityConfig

config = OptimizationConfig(
    use_gpu=True,
    batch_size=64,
    enable_caching=True
)
```

## Development and Contribution

### Code Standards
- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings
- **Testing**: 85%+ code coverage requirement
- **Style**: Black formatting with flake8 linting

### Development Workflow
1. **Fork and branch** for feature development
2. **Implement with tests** following TDD principles
3. **Run quality gates** before submission
4. **Update documentation** for new features

## Performance Benchmarks

| Component | Performance | Memory | Notes |
|-----------|-------------|--------|-------|
| Basic Agent | ~5ms inference | <10MB | Single-threaded |
| Optimized Agent | <1ms inference | <5MB | GPU acceleration |
| Distributed Cluster | ~2ms inference | <50MB | 4-node cluster |
| Batch Processing | 0.1ms per sample | <20MB | 100-sample batches |

## Integration Examples

### Research Workflow
```python
# Setup
from active_inference import create_research_agent

agent = create_research_agent(
    algorithm="hierarchical_active_inference",
    environment="complex_grid_world"
)

# Training loop
for episode in range(1000):
    observation = env.reset()
    done = False

    while not done:
        action = agent.act(observation)
        next_obs, reward, done, info = env.step(action)
        agent.update(observation, action, reward, next_obs)
        observation = next_obs

    # Research metrics
    metrics = agent.get_research_metrics()
    logger.info(f"Episode {episode}: {metrics}")
```

### Production Deployment
```python
# Setup
from active_inference import create_production_agent

agent = create_production_agent(
    optimization_level="high",
    monitoring_enabled=True,
    security_level="enterprise"
)

# Production loop with monitoring
while True:
    try:
        observation = get_observation()
        action = agent.act(observation)
        execute_action(action)

        # Health check
        if agent.health_check():
            continue
        else:
            # Trigger recovery
            agent.recover()

    except Exception as e:
        logger.error(f"Production error: {e}")
        agent.handle_error(e)
```

