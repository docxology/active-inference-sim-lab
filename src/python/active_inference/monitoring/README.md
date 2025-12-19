# Monitoring and Telemetry

The `monitoring` module provides comprehensive observability capabilities for active inference systems, enabling real-time performance tracking, anomaly detection, and health monitoring.

## Core Components

### TelemetryCollector

Centralized metrics collection and distributed tracing.

**Features:**
- Real-time metrics collection with tagging
- Distributed tracing support
- Multiple output formats (JSON, Prometheus, Jaeger)
- Performance-optimized data structures
- Automatic metrics aggregation

**Usage:**
```python
from active_inference.monitoring import TelemetryCollector

# Initialize telemetry
telemetry = TelemetryCollector(
    enable_prometheus=True,
    enable_jaeger=True
)

# Record metrics
telemetry.record_metric(
    name="inference_time",
    value=0.045,
    tags={"agent_id": "agent_001", "model_type": "variational"},
    metric_type="histogram"
)

# Start trace
trace_id = telemetry.start_trace(
    operation_name="belief_update",
    tags={"agent_id": "agent_001"}
)

# ... perform operation ...

# End trace
telemetry.finish_trace(trace_id, result="success")
```

### PerformanceProfiler

Detailed performance analysis and profiling tools.

**Features:**
- Function-level execution time tracking
- Memory usage analysis
- CPU utilization monitoring
- Statistical analysis of performance data
- Context manager for easy profiling

**Usage:**
```python
from active_inference.monitoring import PerformanceProfiler

profiler = PerformanceProfiler("belief_updater")

# Profile function execution
with profiler.profile_function("update_beliefs"):
    beliefs = agent.inference.update_beliefs(prior, observation, model)

# Get performance report
report = profiler.get_performance_report()
print(f"Average execution time: {report['avg_execution_time']:.3f}s")
print(f"Memory delta: {report['avg_memory_delta']:.1f}MB")
```

### AnomalyDetector

Machine learning-based anomaly detection for system monitoring.

**Features:**
- Statistical anomaly detection (Z-score, IQR)
- ML-based anomaly detection (isolation forest)
- Behavioral pattern analysis
- Adaptive threshold adjustment
- Real-time alerting capabilities

**Usage:**
```python
from active_inference.monitoring import AnomalyDetector

detector = AnomalyDetector(methods=['statistical', 'ml', 'behavioral'])

# Check for anomalies
anomalies = detector.detect_threats(
    client_id="agent_001",
    input_data=observation,
    processing_time=0.045
)

if anomalies:
    for anomaly in anomalies:
        print(f"Anomaly detected: {anomaly.description}")
        # Trigger alerts, logging, etc.
```

### HealthMonitor

System health monitoring and automated health checks.

**Features:**
- Component availability monitoring
- Performance threshold checking
- Resource usage tracking
- Automated health reporting
- Integration with alerting systems

**Usage:**
```python
from active_inference.monitoring import HealthMonitor

monitor = HealthMonitor(check_interval=30.0)

# Register components to monitor
monitor.register_component(
    name="inference_engine",
    health_check=lambda: agent.inference.is_healthy()
)

monitor.register_component(
    name="memory_system",
    health_check=lambda: check_memory_usage()
)

# Get overall health status
health_status = monitor.get_system_health()
print(f"System health: {health_status['overall_status']}")

if health_status['overall_status'] != 'healthy':
    print("Unhealthy components:")
    for comp, status in health_status['component_statuses'].items():
        if not status['healthy']:
            print(f"  {comp}: {status['issues']}")
```

## Integration with Active Inference

### Monitored Agent

Active inference agents with built-in monitoring:

```python
from active_inference.monitoring import MonitoredActiveInferenceAgent

# Create monitored agent
agent = MonitoredActiveInferenceAgent(
    state_dim=16,
    obs_dim=32,
    action_dim=4,
    monitoring_config={
        'telemetry_enabled': True,
        'performance_profiling': True,
        'anomaly_detection': True,
        'health_monitoring': True
    }
)

# Normal agent usage - monitoring happens automatically
for episode in range(100):
    observation = env.reset()
    total_reward = 0

    while True:
        action = agent.act(observation)
        next_obs, reward, done, info = env.step(action)
        agent.update(observation, action, reward, next_obs)

        total_reward += reward
        if done:
            break
        observation = next_obs

    # Get monitoring data
    metrics = agent.get_monitoring_data()
    health = agent.get_health_status()

    print(f"Episode {episode}: Reward = {total_reward}")
    print(f"Inference time: {metrics['avg_inference_time']:.3f}s")
    print(f"Health status: {health['overall_status']}")
```

### Custom Metrics

Define and collect custom metrics:

```python
from active_inference.monitoring import TelemetryCollector
from prometheus_client import Counter, Histogram, Gauge

class CustomMetricsCollector:
    """Custom metrics for research experiments."""

    def __init__(self):
        self.telemetry = TelemetryCollector()

        # Define custom metrics
        self.experiment_trials = Counter(
            'experiment_trials_total',
            'Total number of experiment trials',
            ['experiment_name', 'condition']
        )

        self.learning_progress = Gauge(
            'learning_progress',
            'Learning progress metric',
            ['agent_id', 'metric_type']
        )

        self.free_energy_trajectory = Histogram(
            'free_energy_trajectory',
            'Free energy values over time',
            ['agent_id']
        )

    def record_experiment_trial(self, experiment_name, condition, success):
        """Record experiment trial."""

        self.experiment_trials.labels(
            experiment_name=experiment_name,
            condition=condition
        ).inc()

        # Record to telemetry
        self.telemetry.record_metric(
            'experiment_trial',
            1,
            tags={
                'experiment_name': experiment_name,
                'condition': condition,
                'success': success
            }
        )

    def update_learning_progress(self, agent_id, metric_type, value):
        """Update learning progress metric."""

        self.learning_progress.labels(
            agent_id=agent_id,
            metric_type=metric_type
        ).set(value)

    def record_free_energy(self, agent_id, free_energy):
        """Record free energy value."""

        self.free_energy_trajectory.labels(agent_id=agent_id).observe(free_energy)
```

## Monitoring Configuration

### Configuration Options

```python
monitoring_config = {
    # Telemetry settings
    'telemetry': {
        'enabled': True,
        'prometheus_export': True,
        'jaeger_tracing': True,
        'metrics_retention_days': 30,
        'batch_size': 100
    },

    # Performance profiling
    'profiling': {
        'enabled': True,
        'functions_to_profile': ['act', 'update', 'plan_action'],
        'memory_tracking': True,
        'cpu_tracking': True,
        'sampling_rate': 0.1  # Sample 10% of calls
    },

    # Anomaly detection
    'anomaly_detection': {
        'enabled': True,
        'methods': ['statistical', 'ml'],
        'sensitivity': 'medium',  # low, medium, high
        'adaptation_rate': 0.01
    },

    # Health monitoring
    'health_monitoring': {
        'enabled': True,
        'check_interval_seconds': 30,
        'alert_thresholds': {
            'memory_usage_percent': 80,
            'cpu_usage_percent': 70,
            'error_rate': 0.05
        }
    },

    # Alerting
    'alerting': {
        'enabled': True,
        'channels': ['email', 'slack', 'webhook'],
        'cooldown_minutes': 5
    }
}
```

### Environment-Based Configuration

```bash
# Monitoring settings
ACTIVE_INFERENCE_TELEMETRY_ENABLED=true
ACTIVE_INFERENCE_PROMETHEUS_ENABLED=true
ACTIVE_INFERENCE_JAEGER_ENABLED=true

# Performance profiling
ACTIVE_INFERENCE_PROFILING_ENABLED=true
ACTIVE_INFERENCE_MEMORY_TRACKING=true

# Anomaly detection
ACTIVE_INFERENCE_ANOMALY_DETECTION_ENABLED=true
ACTIVE_INFERENCE_ANOMALY_SENSITIVITY=medium

# Health monitoring
ACTIVE_INFERENCE_HEALTH_CHECK_INTERVAL=30
ACTIVE_INFERENCE_MEMORY_ALERT_THRESHOLD=80

# Alerting
ACTIVE_INFERENCE_ALERTING_ENABLED=true
ACTIVE_INFERENCE_SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## Metrics and Dashboards

### Key Metrics

**Performance Metrics:**
- `active_inference_inference_time_seconds`: Time for belief inference
- `active_inference_planning_time_seconds`: Time for action planning
- `active_inference_memory_usage_bytes`: Memory consumption
- `active_inference_cpu_usage_percent`: CPU utilization

**Quality Metrics:**
- `active_inference_prediction_error`: Prediction accuracy
- `active_inference_free_energy`: Free energy levels
- `active_inference_cache_hit_ratio`: Cache performance
- `active_inference_errors_total`: Error counts by type

**System Metrics:**
- `active_inference_agents_active`: Number of active agents
- `active_inference_requests_total`: Total requests processed
- `active_inference_response_time_seconds`: Response time distribution

### Grafana Dashboard

Example dashboard configuration:

```json
{
  "dashboard": {
    "title": "Active Inference Monitoring",
    "panels": [
      {
        "title": "Inference Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(active_inference_inference_time_seconds_bucket[5m]))",
            "legendFormat": "95th Percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "bargauge",
        "targets": [
          {
            "expr": "active_inference_memory_usage_bytes / active_inference_memory_limit_bytes",
            "legendFormat": "Memory Usage %"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(active_inference_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Alerting Rules

### Prometheus Alerting

```yaml
# alerts.yml
groups:
  - name: active-inference
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(active_inference_inference_time_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "95th percentile inference latency > 1.0s for 5 minutes"

      - alert: MemoryUsageCritical
        expr: active_inference_memory_usage_bytes > 1e9
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Critical memory usage"
          description: "Memory usage exceeds 1GB for 3 minutes"

      - alert: HighErrorRate
        expr: rate(active_inference_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate exceeds 10% for 5 minutes"
```

## Testing and Validation

### Monitoring Tests

```python
import pytest
from active_inference.monitoring import TelemetryCollector, PerformanceProfiler

class TestMonitoring:
    def test_telemetry_collection(self):
        """Test telemetry data collection."""
        telemetry = TelemetryCollector()

        # Record metrics
        telemetry.record_metric("test_metric", 42.0, {"tag": "value"})

        metrics = telemetry.get_metrics()
        assert "test_metric" in metrics
        assert len(metrics["test_metric"]) == 1

    def test_performance_profiling(self):
        """Test performance profiling."""
        profiler = PerformanceProfiler("test_component")

        with profiler.profile_function("test_function"):
            import time
            time.sleep(0.01)  # Simulate work

        report = profiler.get_performance_report()
        assert "test_function" in report
        assert report["test_function"]["call_count"] == 1
        assert report["test_function"]["execution_time"] > 0

    def test_health_monitoring(self):
        """Test health monitoring."""
        monitor = HealthMonitor()

        # Register healthy component
        monitor.register_component("test_comp", lambda: {"healthy": True})

        health = monitor.get_system_health()
        assert health["overall_status"] == "healthy"
        assert "test_comp" in health["component_statuses"]

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        detector = AnomalyDetector()

        # Normal behavior
        normal_anomalies = detector.detect_threats("agent_1", [1.0, 2.0], 0.1)
        assert len(normal_anomalies) == 0

        # Anomalous behavior (very slow processing)
        anomalies = detector.detect_threats("agent_1", [1.0, 2.0], 5.0)
        assert len(anomalies) > 0
```

## Troubleshooting

### Common Monitoring Issues

**Metrics Not Appearing:**
```python
# Check telemetry configuration
telemetry = TelemetryCollector()
print(f"Prometheus enabled: {telemetry.enable_prometheus}")

# Verify metric recording
telemetry.record_metric("test", 1.0)
metrics = telemetry.get_metrics()
print(f"Metrics recorded: {list(metrics.keys())}")
```

**High Memory Usage:**
```python
# Check metrics retention settings
telemetry = TelemetryCollector(max_metrics_per_type=1000)

# Monitor memory usage
profiler = PerformanceProfiler("memory_check")
with profiler.profile_function("memory_test"):
    # Your code here

report = profiler.get_performance_report()
print(f"Memory usage: {report['memory_delta']:.1f}MB")
```

**False Positive Anomalies:**
```python
# Adjust anomaly detection sensitivity
detector = AnomalyDetector(sensitivity="low")

# Or customize detection methods
detector = AnomalyDetector(methods=['statistical'])  # Disable ML detection
```

## Performance Considerations

### Optimization Strategies

**Batch Metric Collection:**
```python
# Efficient batch processing
metrics_batch = []
for i in range(1000):
    metrics_batch.append({
        'name': f'metric_{i}',
        'value': i * 0.1,
        'tags': {'batch': 'test'}
    })

# Batch record
telemetry.record_metrics_batch(metrics_batch)
```

**Sampling for High-Frequency Metrics:**
```python
import random

# Sample 10% of high-frequency operations
if random.random() < 0.1:
    telemetry.record_metric("high_freq_operation", value, tags)
```

**Asynchronous Monitoring:**
```python
import threading

def background_monitoring():
    """Run monitoring in background thread."""
    while True:
        # Collect metrics
        metrics = collect_system_metrics()

        # Record asynchronously
        telemetry.record_metric("system_health", metrics['health_score'])

        time.sleep(60)  # Every minute

# Start background monitoring
monitor_thread = threading.Thread(target=background_monitoring, daemon=True)
monitor_thread.start()
```

## Integration Examples

### Research Experiment Monitoring

```python
from active_inference.monitoring import TelemetryCollector, ExperimentTracker

# Setup experiment monitoring
telemetry = TelemetryCollector()
experiment_tracker = ExperimentTracker("hierarchical_ai_experiment")

# Configure metrics
experiment_tracker.add_metric("free_energy", "gauge")
experiment_tracker.add_metric("learning_progress", "gauge")
experiment_tracker.add_metric("inference_time", "histogram")

# Run experiment with monitoring
for episode in range(1000):
    # Training logic
    free_energy = agent.get_free_energy()
    learning_progress = calculate_learning_progress()

    # Record metrics
    experiment_tracker.record_metric("free_energy", free_energy)
    experiment_tracker.record_metric("learning_progress", learning_progress)

    with experiment_tracker.time_operation("inference"):
        action = agent.act(observation)

    # Log progress
    if episode % 100 == 0:
        metrics = experiment_tracker.get_summary_metrics()
        print(f"Episode {episode}: Free Energy = {metrics['free_energy']['mean']:.3f}")

# Generate experiment report
report = experiment_tracker.generate_report()
report.save("experiment_results.json")
```

### Production System Monitoring

```python
from active_inference.monitoring import ProductionMonitor

# Setup production monitoring
monitor = ProductionMonitor(
    alert_webhook_url="https://hooks.slack.com/...",
    metrics_retention_days=30
)

# Register components
monitor.register_component("api_server", check_api_health)
monitor.register_component("database", check_database_health)
monitor.register_component("cache", check_cache_health)

# Setup alerting
monitor.setup_alerts({
    'memory_usage': {'threshold': 80, 'severity': 'warning'},
    'error_rate': {'threshold': 0.05, 'severity': 'critical'},
    'response_time': {'threshold': 1.0, 'severity': 'warning'}
})

# Monitor in production loop
while True:
    try:
        # Process requests
        response = handle_request(request)

        # Record metrics
        monitor.record_request(
            response_time=response.processing_time,
            success=response.success,
            error_type=response.error_type if not response.success else None
        )

    except Exception as e:
        monitor.record_error("request_processing", str(e))

    # Periodic health checks
    if time.time() % 60 < 1:  # Every ~60 seconds
        health_status = monitor.check_system_health()
        if not health_status['healthy']:
            monitor.trigger_alert("system_unhealthy", health_status)
```

## Contributing

### Adding New Metrics

1. **Define Metric:**
   ```python
   from active_inference.monitoring import TelemetryCollector

   # Add to telemetry
   telemetry.add_custom_metric(
       name="custom_metric",
       type="gauge",
       description="Custom metric description",
       labels=["agent_id", "component"]
   )
   ```

2. **Record Metric:**
   ```python
   telemetry.record_metric(
       "custom_metric",
       value=42.0,
       tags={"agent_id": "agent_1", "component": "inference"}
   )
   ```

3. **Add to Dashboards:**
   Update Grafana dashboards to include the new metric.

### Extending Monitoring

1. **Create Custom Monitor:**
   ```python
   from active_inference.monitoring import BaseMonitor

   class CustomMonitor(BaseMonitor):
       def collect_metrics(self):
           # Custom metric collection logic
           return {"custom_metric": get_custom_value()}

       def check_health(self):
           # Custom health check logic
           return is_system_healthy()
   ```

2. **Integrate with Framework:**
   ```python
   # Register custom monitor
   monitoring_system.register_monitor(CustomMonitor())
   ```

3. **Add Tests:**
   ```python
   def test_custom_monitor():
       monitor = CustomMonitor()
       metrics = monitor.collect_metrics()
       assert "custom_metric" in metrics

       health = monitor.check_health()
       assert isinstance(health, bool)
   ```

## References

- [Prometheus Monitoring](https://prometheus.io/)
- [Grafana Visualization](https://grafana.com/)
- [OpenTelemetry](https://opentelemetry.io/)
- [Monitoring Distributed Systems](https://www.oreilly.com/library/view/monitoring-distributed-systems/)

