# Examples & Use Cases - AGENTS

## Module Overview

The `examples` directory contains practical code examples and use cases demonstrating the Active Inference Simulation Lab's capabilities across research, development, and production scenarios.

## Example Categories

### Basic Usage Examples

**Core Agent Creation:**
```python
"""
Basic Active Inference Agent Example

Demonstrates the fundamental usage of active inference agents.
"""

from active_inference import ActiveInferenceAgent
import numpy as np

def basic_agent_example():
    """Create and use a basic active inference agent."""

    # Initialize agent
    agent = ActiveInferenceAgent(
        state_dim=4,      # Hidden state dimensionality
        obs_dim=8,        # Observation dimensionality
        action_dim=2,     # Action dimensionality
        inference_method='variational',
        planning_horizon=5
    )

    # Example observation
    observation = np.random.randn(8)

    # Get action from agent
    action = agent.act(observation)

    print(f"Agent selected action: {action}")
    print(f"Agent beliefs: {agent.get_beliefs()}")

    return agent

if __name__ == "__main__":
    basic_agent_example()
```

**Environment Integration:**
```python
"""
Environment Integration Example

Shows how to integrate active inference agents with various environments.
"""

from active_inference import ActiveInferenceAgent
from active_inference.environments import GridWorld, GymWrapper
import gymnasium as gym

def grid_world_example():
    """Active inference agent in a grid world."""

    # Create grid world environment
    env = GridWorld(
        size=(10, 10),
        obstacles=[(3, 3), (7, 7)],
        goals=[(9, 9)],
        partial_observability=True,
        observation_radius=2
    )

    # Create agent
    agent = ActiveInferenceAgent(
        state_dim=4,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )

    # Run episode
    observation = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(observation)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward

        # Agent learns from experience
        agent.update(observation, action, reward, next_obs)
        observation = next_obs

    print(f"Episode completed with total reward: {total_reward}")

def gym_integration_example():
    """Integration with OpenAI Gym environments."""

    # Wrap Gym environment
    gym_env = gym.make("CartPole-v1")
    env = GymWrapper(gym_env, add_model_uncertainty=True)

    # Create agent
    agent = ActiveInferenceAgent.from_env(env)

    # Training loop
    num_episodes = 100

    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(observation)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward

            agent.update(observation, action, reward, next_obs)
            observation = next_obs

        print(f"Episode {episode}: Reward = {episode_reward}")

if __name__ == "__main__":
    grid_world_example()
    gym_integration_example()
```

### Research Examples

**Hierarchical Active Inference:**
```python
"""
Hierarchical Active Inference Example

Demonstrates multi-timescale hierarchical active inference.
"""

from active_inference.research import HierarchicalTemporalActiveInference
import numpy as np

def hierarchical_agent_example():
    """Create and train a hierarchical active inference agent."""

    # Create hierarchical agent with 3 levels
    agent = HierarchicalTemporalActiveInference(
        n_levels=3,
        temporal_scales=[1, 5, 15],  # Different timescales for each level
        state_dims=[4, 8, 16],       # Increasing complexity at higher levels
        obs_dim=8,
        action_dim=2
    )

    # Training data
    observations = [np.random.randn(8) for _ in range(1000)]

    print("Training hierarchical agent...")

    for i, obs in enumerate(observations):
        # Hierarchical processing
        action = agent.act(obs)

        # Simulate environment response
        reward = np.random.randn()  # Random reward for demonstration
        next_obs = np.random.randn(8)

        # Update all levels of hierarchy
        agent.update(obs, action, reward, next_obs)

        if i % 100 == 0:
            print(f"Processed {i} observations")

            # Analyze hierarchical beliefs
            beliefs = agent.get_hierarchical_beliefs()
            for level, level_beliefs in enumerate(beliefs):
                print(f"Level {level} beliefs shape: {level_beliefs.shape}")

    # Analyze learned hierarchies
    hierarchy_analysis = agent.analyze_hierarchy()
    print("Hierarchy analysis:")
    print(f"Temporal abstraction: {hierarchy_analysis['temporal_abstraction']}")
    print(f"Cross-level information flow: {hierarchy_analysis['information_flow']}")

if __name__ == "__main__":
    hierarchical_agent_example()
```

**Causal Active Inference:**
```python
"""
Causal Active Inference Example

Demonstrates causal reasoning and interventional planning.
"""

from active_inference.research import CausalActiveInference
import numpy as np

def causal_agent_example():
    """Causal active inference for interventional planning."""

    # Create causal agent
    agent = CausalActiveInference(
        base_agent=None,  # Will create default agent
        intervention_budget=3,
        causal_graph_learning=True
    )

    # Define causal scenario (treatment effect estimation)
    # Simulate a causal system: Treatment → Outcome, Confounder → Both
    np.random.seed(42)

    # Generate synthetic data with causal relationships
    n_samples = 1000
    confounder = np.random.randn(n_samples)
    treatment = confounder + 0.5 * np.random.randn(n_samples)
    outcome = 2.0 * treatment + confounder + 0.3 * np.random.randn(n_samples)

    print("Learning causal relationships...")

    for i in range(n_samples):
        # Create observation from causal variables
        observation = np.array([treatment[i], outcome[i], confounder[i]])

        # Agent observes and learns causal structure
        agent.causal_reasoning_and_planning(
            observation,
            goal_state=np.array([1.0, 2.0, 0.0]),  # Desired outcome
            intervention_budget=3
        )

    # Analyze learned causal model
    causal_stats = agent.get_causal_statistics()

    print("Causal model learned:")
    print(f"Discovered relations: {causal_stats['causal_graph']['edges']}")
    print(f"Causal strength: {causal_stats['causal_graph']['avg_edge_weight']:.3f}")
    print(f"Intervention success rate: {causal_stats['interventions']['successful_interventions'] / max(1, causal_stats['interventions']['total_interventions']):.2%}")

    # Demonstrate interventional planning
    print("\\nDemonstrating interventional planning...")

    # Current state: low treatment, low outcome
    current_state = np.array([0.0, 0.0, 0.0])
    desired_state = np.array([1.5, 3.0, 0.0])  # High treatment, high outcome

    intervention_plan = agent.causal_reasoning_and_planning(
        current_state,
        desired_state,
        intervention_budget=2
    )

    print("Intervention plan:")
    print(f"Selected interventions: {intervention_plan['intervention_plan']['selected_interventions']}")
    print(f"Expected causal effect: {intervention_plan['counterfactual_analysis']['intervention_effects']}")

if __name__ == "__main__":
    causal_agent_example()
```

### Performance Optimization Examples

**GPU Acceleration:**
```python
"""
GPU-Accelerated Active Inference Example

Demonstrates GPU acceleration for high-performance active inference.
"""

from active_inference.performance import PerformanceOptimizedActiveInferenceAgent, GPUAccelerator
import torch
import time
import numpy as np

def gpu_accelerated_example():
    """GPU-accelerated active inference processing."""

    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return

    print(f"GPU available: {torch.cuda.get_device_name()}")

    # Create GPU accelerator
    gpu_accelerator = GPUAccelerator(enable_cuda=True)

    # Create performance-optimized agent
    agent = PerformanceOptimizedActiveInferenceAgent(
        state_dim=64,
        obs_dim=128,
        action_dim=8,
        optimization_config={
            'use_gpu': True,
            'batch_size': 32,
            'enable_caching': True,
            'memory_pool_size': 1024
        }
    )

    # Generate batch of observations
    batch_size = 100
    obs_dim = 128
    observations = [np.random.randn(obs_dim) for _ in range(batch_size)]

    print(f"Processing {batch_size} observations...")

    # Time GPU-accelerated processing
    start_time = time.time()

    actions = []
    for obs in observations:
        action = agent.act(obs)
        actions.append(action)

    gpu_time = time.time() - start_time

    print(".3f")
    print(".1f")
    print(f"GPU utilization: {gpu_accelerator.get_performance_stats()['compute_utilization']['cuda']}%")

    # Compare with CPU version (simulated)
    print("\\nComparing with CPU baseline...")

    cpu_agent = PerformanceOptimizedActiveInferenceAgent(
        state_dim=64,
        obs_dim=128,
        action_dim=8,
        optimization_config={
            'use_gpu': False,
            'batch_size': 1,
            'enable_caching': False
        }
    )

    start_time = time.time()
    cpu_actions = []
    for obs in observations[:10]:  # Test subset for CPU
        action = cpu_agent.act(obs)
        cpu_actions.append(action)

    cpu_time = time.time() - start_time

    # Extrapolate CPU time for full batch
    extrapolated_cpu_time = cpu_time * (batch_size / 10)

    speedup = extrapolated_cpu_time / gpu_time
    print(".1f")
    print(".2f")

if __name__ == "__main__":
    gpu_accelerated_example()
```

**Batch Processing:**
```python
"""
Batch Processing Example

Demonstrates efficient batch processing for high-throughput scenarios.
"""

from active_inference.performance import ParallelProcessor
from active_inference import ActiveInferenceAgent
import numpy as np
import time

def batch_processing_example():
    """Batch processing for high-throughput active inference."""

    # Create multiple agents
    num_agents = 10
    agents = []

    for i in range(num_agents):
        agent = ActiveInferenceAgent(
            state_dim=8,
            obs_dim=16,
            action_dim=4,
            agent_id=f"agent_{i}"
        )
        agents.append(agent)

    # Create parallel processor
    processor = ParallelProcessor(max_workers=4)

    # Generate batch observations for all agents
    batch_size = 50
    obs_dim = 16

    print(f"Processing {batch_size} observations for {num_agents} agents...")

    start_time = time.time()

    # Process in parallel
    results = []
    for batch_idx in range(batch_size):
        # Generate observations for all agents
        batch_observations = [np.random.randn(obs_dim) for _ in range(num_agents)]

        # Parallel processing
        batch_results = processor.parallel_inference(agents, batch_observations)
        results.extend(batch_results)

    processing_time = time.time() - start_time

    throughput = (batch_size * num_agents) / processing_time

    print(".3f")
    print(".1f")
    print(f"Parallel efficiency: {processor.get_parallel_stats()['cpu_count'] / processor.max_workers:.2f}")

    # Analyze results
    successful_actions = sum(1 for r in results if r is not None)
    success_rate = successful_actions / len(results)

    print(f"Success rate: {success_rate:.2%}")

    # Performance breakdown
    stats = processor.get_parallel_stats()
    print(f"Active threads: {stats['active_threads']}")
    print(f"CPU cores utilized: {stats['cpu_count']}")

if __name__ == "__main__":
    batch_processing_example()
```

### Production Deployment Examples

**Microservice Architecture:**
```python
"""
Microservice Deployment Example

Demonstrates active inference in a microservice architecture.
"""

from flask import Flask, request, jsonify
from active_inference import ActiveInferenceAgent
from active_inference.monitoring import TelemetryCollector
from active_inference.security import SecurityValidator
import threading
import time

app = Flask(__name__)

class ActiveInferenceService:
    """Active inference as a microservice."""

    def __init__(self):
        # Initialize components
        self.agent = ActiveInferenceAgent(
            state_dim=16,
            obs_dim=32,
            action_dim=8
        )

        self.telemetry = TelemetryCollector()
        self.security = SecurityValidator()

        # Start background monitoring
        self.monitoring_thread = threading.Thread(
            target=self._background_monitoring,
            daemon=True
        )
        self.monitoring_thread.start()

    def _background_monitoring(self):
        """Background health and performance monitoring."""

        while True:
            # Collect metrics
            memory_usage = self._get_memory_usage()
            self.telemetry.record_metric(
                'service_memory_usage',
                memory_usage,
                {'service': 'active_inference'}
            )

            # Health check
            health_status = self._check_health()
            self.telemetry.record_metric(
                'service_health',
                1 if health_status else 0,
                {'service': 'active_inference'}
            )

            time.sleep(30)  # Monitor every 30 seconds

    def _get_memory_usage(self):
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def _check_health(self):
        """Check service health."""
        try:
            # Quick health check
            test_obs = [0.0] * 32
            action = self.agent.act(test_obs)
            return action is not None
        except Exception:
            return False

    def process_request(self, observation_data, user_context=None):
        """Process active inference request."""

        start_time = time.time()

        try:
            # Security validation
            validation_result = self.security.validate_secure(
                observation_data, 'observation', context=user_context
            )

            if not validation_result.is_valid:
                return {
                    'status': 'error',
                    'message': 'Security validation failed',
                    'errors': validation_result.errors
                }

            # Process with agent
            observation = observation_data['observation']
            action = self.agent.act(observation)

            # Record telemetry
            processing_time = time.time() - start_time
            self.telemetry.record_metric(
                'request_processing_time',
                processing_time,
                {'endpoint': 'process_request'}
            )

            return {
                'status': 'success',
                'action': action.tolist(),
                'processing_time': processing_time,
                'agent_beliefs': self.agent.get_beliefs()
            }

        except Exception as e:
            # Record error
            self.telemetry.record_metric(
                'request_errors',
                1,
                {'error_type': type(e).__name__}
            )

            return {
                'status': 'error',
                'message': str(e)
            }

# Global service instance
service = ActiveInferenceService()

@app.route('/health')
def health():
    """Health check endpoint."""
    is_healthy = service._check_health()
    status_code = 200 if is_healthy else 503

    return jsonify({
        'healthy': is_healthy,
        'timestamp': time.time()
    }), status_code

@app.route('/api/v1/act', methods=['POST'])
def act():
    """Active inference action endpoint."""

    try:
        data = request.get_json()

        if not data or 'observation' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing observation data'
            }), 400

        # Process request
        result = service.process_request(data, user_context=request.headers)

        status_code = 200 if result['status'] == 'success' else 400

        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest
    return generate_latest()

if __name__ == "__main__":
    print("Starting Active Inference Microservice...")
    print("Health check: http://localhost:8000/health")
    print("API endpoint: http://localhost:8000/api/v1/act")
    print("Metrics: http://localhost:8000/metrics")

    app.run(host='0.0.0.0', port=8000, debug=False)
```

**Kubernetes Deployment:**
```yaml
# examples/kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: active-inference-service
  labels:
    app: active-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: active-inference
  template:
    metadata:
      labels:
        app: active-inference
    spec:
      containers:
      - name: active-inference
        image: active-inference:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYTHONPATH
          value: "/app/src/python"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: active-inference-service
spec:
  selector:
    app: active-inference
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: active-inference-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: ai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: active-inference-service
            port:
              number: 80
```

### Advanced Research Examples

**Meta-Learning Active Inference:**
```python
"""
Meta-Learning Active Inference Example

Demonstrates meta-learning capabilities for rapid adaptation.
"""

from active_inference.research import MetaActiveInference
import numpy as np

def meta_learning_example():
    """Meta-learning active inference for few-shot adaptation."""

    # Create meta-learning agent
    meta_agent = MetaActiveInference(
        base_agent_config={
            'state_dim': 8,
            'obs_dim': 16,
            'action_dim': 4
        },
        meta_learning_rate=0.01,
        adaptation_steps=5
    )

    # Define different tasks (different reward functions)
    tasks = [
        {'name': 'approach_target', 'reward_func': lambda obs, action: -np.linalg.norm(obs[:2])},
        {'name': 'avoid_obstacle', 'reward_func': lambda obs, action: -np.linalg.norm(obs[2:4])},
        {'name': 'maintain_velocity', 'reward_func': lambda obs, action: -abs(obs[4] - 1.0)},
    ]

    print("Meta-learning across tasks...")

    for task_idx, task in enumerate(tasks):
        print(f"\\nAdapting to task: {task['name']}")

        # Meta-learning phase: adapt to new task
        meta_agent.begin_task_adaptation(task)

        # Few-shot learning within task
        task_performance = []

        for episode in range(10):  # Few-shot learning
            episode_reward = 0

            for step in range(50):
                # Generate observation (simplified)
                observation = np.random.randn(16)

                # Get action with meta-learned adaptation
                action = meta_agent.act_adapted(observation, task)

                # Get reward for this task
                reward = task['reward_func'](observation, action)
                episode_reward += reward

                # Update meta-knowledge
                meta_agent.update_meta_knowledge(observation, action, reward)

            task_performance.append(episode_reward)
            print(f"Episode {episode}: Reward = {episode_reward:.2f}")

        avg_performance = np.mean(task_performance)
        print(f"Average performance on {task['name']}: {avg_performance:.2f}")

        # Analyze meta-learning effectiveness
        meta_stats = meta_agent.get_meta_statistics()
        print(f"Meta-learning stats: {meta_stats}")

    # Final evaluation across all tasks
    print("\\nEvaluating meta-learned capabilities...")

    for task in tasks:
        # Test without adaptation (using meta-knowledge)
        test_performance = []

        for episode in range(5):
            episode_reward = 0

            for step in range(20):
                observation = np.random.randn(16)
                action = meta_agent.act_with_meta_knowledge(observation, task)
                reward = task['reward_func'](observation, action)
                episode_reward += reward

            test_performance.append(episode_reward)

        avg_test_performance = np.mean(test_performance)
        print(f"Meta-knowledge performance on {task['name']}: {avg_test_performance:.2f}")

if __name__ == "__main__":
    meta_learning_example()
```

## Example Organization

### Directory Structure

```
examples/
├── basic/                 # Basic usage examples
│   ├── simple_agent.py
│   ├── environment_integration.py
│   └── training_loops.py
├── research/              # Research-oriented examples
│   ├── hierarchical_ai.py
│   ├── causal_inference.py
│   ├── meta_learning.py
│   └── continual_learning.py
├── performance/           # Performance optimization examples
│   ├── gpu_acceleration.py
│   ├── batch_processing.py
│   ├── distributed_computing.py
│   └── memory_optimization.py
├── production/            # Production deployment examples
│   ├── microservice.py
│   ├── kubernetes_deployment.yaml
│   ├── docker_compose.yml
│   └── monitoring_setup.py
├── tutorials/             # Step-by-step tutorials
│   ├── getting_started/
│   ├── advanced_features/
│   └── production_deployment/
└── benchmarks/            # Performance benchmarking examples
    ├── scalability_tests.py
    ├── accuracy_benchmarks.py
    └── comparative_analysis.py
```

### Example Metadata

Each example includes comprehensive metadata:

```python
"""
Active Inference Example: Basic Agent Creation

Description:
    This example demonstrates the fundamental steps for creating and using
    an active inference agent in the Active Inference Simulation Lab.

Level: Beginner
Topics: Agent Creation, Basic Usage, Environment Integration
Prerequisites: Python basics, numpy familiarity
Estimated Time: 15 minutes

Learning Objectives:
    - Understand the basic ActiveInferenceAgent interface
    - Learn how to configure agent parameters
    - Practice basic agent-environment interaction
    - Analyze agent beliefs and actions

Related Examples:
    - environment_integration.py: Advanced environment usage
    - training_loops.py: Training and evaluation patterns
    - gpu_acceleration.py: Performance optimization

Files:
    - basic_agent.py: Main example code
    - README.md: Detailed explanation and walkthrough
    - requirements.txt: Dependencies for this example
"""

# Example code follows...
```

## Running Examples

### Local Execution

```bash
# Install dependencies
pip install -r examples/requirements.txt

# Run basic example
python examples/basic/simple_agent.py

# Run with visualization
python examples/basic/simple_agent.py --visualize

# Run performance benchmark
python examples/benchmarks/scalability_tests.py --agents 100 --episodes 1000
```

### Docker Execution

```bash
# Build example container
docker build -f examples/Dockerfile -t ai-examples .

# Run example
docker run ai-examples python basic/simple_agent.py

# Run with mounted volume for development
docker run -v $(pwd)/examples:/app/examples ai-examples python research/hierarchical_ai.py
```

### Kubernetes Execution

```bash
# Deploy examples to Kubernetes
kubectl apply -f examples/kubernetes/

# Check deployment status
kubectl get pods -l app=ai-examples

# View logs
kubectl logs -l app=ai-examples

# Scale deployment
kubectl scale deployment ai-examples --replicas=5
```

## Example Testing

### Automated Testing

```python
# examples/test_examples.py
import pytest
import subprocess
import sys

def test_basic_examples():
    """Test that basic examples run without errors."""

    examples = [
        'basic/simple_agent.py',
        'basic/environment_integration.py',
        'research/hierarchical_ai.py'
    ]

    for example in examples:
        result = subprocess.run([
            sys.executable, f'examples/{example}', '--test-mode'
        ], capture_output=True, text=True, timeout=60)

        assert result.returncode == 0, f"Example {example} failed: {result.stderr}"
        assert "success" in result.stdout.lower()

def test_performance_examples():
    """Test performance examples meet benchmarks."""

    result = subprocess.run([
        sys.executable, 'examples/benchmarks/scalability_tests.py',
        '--quick-test', '--json-output'
    ], capture_output=True, text=True, timeout=300)

    assert result.returncode == 0

    # Parse JSON output and check benchmarks
    import json
    benchmarks = json.loads(result.stdout)

    assert benchmarks['throughput'] > 1000, "Throughput benchmark failed"
    assert benchmarks['latency_p95'] < 0.1, "Latency benchmark failed"
```

## Contributing Examples

### Example Submission Guidelines

1. **Code Quality**: Follow PEP 8 and include comprehensive docstrings
2. **Documentation**: Include README.md with setup and usage instructions
3. **Testing**: Provide automated tests and expected outputs
4. **Metadata**: Include example metadata and learning objectives
5. **Dependencies**: Specify all required dependencies
6. **Compatibility**: Test on multiple Python versions and platforms

### Example Review Process

1. **Automated Checks**: Linting, type checking, and basic functionality tests
2. **Peer Review**: Technical review by maintainers
3. **Integration Testing**: Full test suite execution
4. **Documentation Review**: Clarity and completeness check
5. **Performance Validation**: Benchmarking against established baselines

## Future Example Enhancements

### Interactive Examples
- **Jupyter Notebooks**: Executable tutorials with visualizations
- **Web Interface**: Browser-based interactive examples
- **Live Demos**: Real-time demonstrations with user input

### Advanced Scenarios
- **Multi-Agent Systems**: Coordination and communication examples
- **Real-World Applications**: Industry-specific use cases
- **Edge Computing**: Resource-constrained deployment examples
- **Federated Learning**: Privacy-preserving distributed examples

### Educational Content
- **Video Tutorials**: Screencast walkthroughs of complex examples
- **Progressive Difficulty**: Examples ordered by complexity
- **Concept Maps**: Visual learning paths through examples
- **Assessment Tools**: Automated evaluation of learning progress

