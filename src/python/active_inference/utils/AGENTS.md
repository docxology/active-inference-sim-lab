# Utility Functions - AGENTS

## Module Overview

The `utils` module provides essential utility functions and infrastructure components that support all other modules in the Active Inference Simulation Lab. These utilities ensure consistent behavior, error handling, validation, logging, and performance monitoring across the entire framework.

## Core Utility Components

### Logging Infrastructure

#### StructuredLogger

Advanced logging system with structured output, performance monitoring, and telemetry collection.

**Features:**
- JSON-structured logging for machine readability
- Performance metrics tracking
- Error aggregation and reporting
- Session-based logging with rotation
- Multiple output destinations (console, file, network)

**Key Classes:**
- `StructuredLogger`: Main logging interface with structured output
- `LogEntry`: Structured log entry with metadata
- `PerformanceTimer`: Context manager for timing operations
- `TelemetryLogger`: Event-based telemetry collection

**Usage:**
```python
from active_inference.utils.logging_config import StructuredLogger, get_logger

# Create structured logger
logger = StructuredLogger(
    "agent_core",
    log_level=LogLevel.INFO,
    log_file="logs/agent.log",
    enable_json=True
)

# Structured logging
logger.info("Agent initialized", LogCategory.AGENT, {
    "agent_id": "agent_001",
    "state_dim": 4,
    "obs_dim": 8
})

# Performance timing
with logger.performance_timer("inference_operation"):
    result = perform_inference(observation)

# Global logger access
main_logger = get_logger("main")
```

#### PerformanceMonitor

Real-time performance monitoring and bottleneck identification.

**Features:**
- Function-level execution time tracking
- Memory usage monitoring
- CPU utilization tracking
- Statistical analysis of performance metrics

**Usage:**
```python
from active_inference.utils.logging_config import PerformanceMonitor

monitor = PerformanceMonitor("belief_updater")

with monitor.measure("update_beliefs"):
    beliefs = update_beliefs(observation, prior_beliefs)

metrics = monitor.get_metrics()
print(f"Average time: {metrics['average_time']:.3f}s")
```

### Validation Framework

#### AdvancedValidator

Comprehensive validation system with security focus and multi-level checking.

**Features:**
- Type validation and bounds checking
- Semantic consistency validation
- Security threat detection
- Performance impact assessment
- Configurable validation strictness

**Key Classes:**
- `ValidationResult`: Structured validation outcome
- `AdvancedValidator`: Main validation engine
- `ValidationException`: Custom validation errors

**Usage:**
```python
from active_inference.utils.advanced_validation import AdvancedValidator

validator = AdvancedValidator(security_level="high")

# Comprehensive validation
result = validator.validate_comprehensive(
    observation_data,
    "observation",
    context={"user_id": "user_123", "session_id": "sess_456"}
)

if not result.is_valid:
    logger.error(f"Validation failed: {result.errors}")
    raise ValidationException(f"Input validation failed", result)
```

#### Input Validation Utilities

Standardized input validation functions for common data types.

**Features:**
- Numpy array validation
- Shape and type checking
- Value range validation
- NaN/inf detection

**Usage:**
```python
from active_inference.utils.advanced_validation import validate_array, validate_inputs

# Array validation
validate_array(observation, "observation", expected_shape=(8,), dtype=np.float64)

# Function parameter validation
@validate_inputs(observation=lambda x: validate_array(x, "observation", (8,)))
def process_observation(self, observation):
    return self.agent.act(observation)
```

### Error Handling System

#### Robust Execution Framework

Decorator-based error handling with automatic recovery strategies.

**Features:**
- Configurable retry mechanisms
- Fallback execution paths
- Error logging and reporting
- Graceful degradation options

**Key Decorators:**
- `@handle_errors`: Basic error handling with logging
- `@robust_execution`: Advanced error handling with retries and fallbacks
- `@circuit_breaker`: Circuit breaker pattern integration

**Usage:**
```python
from active_inference.utils.advanced_validation import handle_errors, robust_execution

@robust_execution(max_retries=3, fallback_value=None)
def unreliable_operation(self, data):
    # Operation that might fail
    if random.random() < 0.3:  # 30% failure rate
        raise ConnectionError("Network timeout")
    return process_data(data)

@handle_errors((ValueError, TypeError), log_errors=True)
def safe_data_processing(self, raw_data):
    # Safe data processing with error handling
    processed = validate_and_clean(raw_data)
    return processed
```

### Caching System

#### Intelligent Caching

Domain-aware caching with multiple strategies and performance optimization.

**Features:**
- LRU (Least Recently Used) eviction
- Domain-specific cache keys
- Cache performance monitoring
- Memory-efficient storage

**Key Classes:**
- `LRUCache`: Basic LRU cache implementation
- `AdaptiveCache`: Intelligent cache with domain awareness
- `CacheManager`: Multi-level cache management

**Usage:**
```python
from active_inference.performance.caching import AdaptiveCache, CacheManager

# Single cache instance
cache = AdaptiveCache(max_size=1000, ttl=300)  # 5 minute TTL

# Cache expensive computation
key = f"belief_update_{hash(str(observation))}"
result = cache.get(key)

if result is None:
    result = expensive_belief_update(observation)
    cache.put(key, result)

# Multi-level caching
cache_manager = CacheManager()
cache_manager.get("belief_state", level="l1")  # Fast access
cache_manager.get("belief_state", level="l3")  # Persistent storage
```

### Security Middleware

#### Security Validation

Input sanitization and security threat detection.

**Features:**
- SQL injection prevention
- XSS attack detection
- Input sanitization
- Security event logging

**Key Functions:**
- `sanitize_input`: Input sanitization and cleaning
- `detect_security_threats`: Threat pattern detection
- `validate_security_context`: Security context validation

**Usage:**
```python
from active_inference.utils.security import sanitize_input, detect_security_threats

# Input sanitization
clean_input = sanitize_input(user_input, input_type="text")

# Threat detection
threats = detect_security_threats(clean_input)
if threats:
    logger.warning(f"Security threats detected: {threats}")
    raise SecurityException("Input contains security threats")
```

### Health Monitoring

#### System Health Checks

Comprehensive health monitoring for all system components.

**Features:**
- Component availability checking
- Performance threshold monitoring
- Resource usage tracking
- Automated health reporting

**Key Classes:**
- `HealthMonitor`: Main health monitoring system
- `HealthCheck`: Individual health check definition
- `HealthReport`: Structured health status reporting

**Usage:**
```python
from active_inference.utils.health_check import HealthMonitor

monitor = HealthMonitor()

# Register health checks
monitor.register_check("database", lambda: check_database_connection())
monitor.register_check("inference_engine", lambda: check_inference_engine())

# Get health status
health_status = monitor.get_health_status()
if not health_status['healthy']:
    logger.error(f"System unhealthy: {health_status['issues']}")
```

### Configuration Management

#### Environment Configuration

Flexible configuration management with environment variable support.

**Features:**
- Environment variable parsing
- Configuration validation
- Default value handling
- Type conversion and validation

**Key Functions:**
- `load_config`: Load configuration from environment
- `validate_config`: Configuration validation
- `get_config_value`: Type-safe configuration access

**Usage:**
```python
from active_inference.utils.config import load_config, get_config_value

# Load configuration
config = load_config(prefix="ACTIVE_INFERENCE_")

# Type-safe access
learning_rate = get_config_value(config, "learning_rate", float, default=0.01)
batch_size = get_config_value(config, "batch_size", int, default=32)
enable_gpu = get_config_value(config, "enable_gpu", bool, default=False)
```

## Advanced Utility Functions

### Mathematical Utilities

Specialized mathematical functions for active inference computations.

**Features:**
- Numerical stability functions
- Statistical computations
- Matrix operations
- Optimization utilities

**Key Functions:**
- `safe_divide`: Numerically stable division
- `stable_softmax`: Numerically stable softmax
- `compute_free_energy`: Free energy computation utilities
- `matrix_sqrt`: Stable matrix square root

### Data Processing Utilities

Data transformation and preprocessing functions.

**Features:**
- Data normalization and standardization
- Missing data handling
- Outlier detection and removal
- Feature engineering utilities

**Key Functions:**
- `normalize_observations`: Observation normalization
- `handle_missing_data`: Missing data imputation
- `detect_outliers`: Statistical outlier detection
- `feature_engineering`: Automatic feature extraction

### Performance Profiling Utilities

Detailed performance analysis and profiling tools.

**Features:**
- Memory profiling
- CPU profiling
- Function call tracing
- Performance bottleneck identification

**Key Classes:**
- `MemoryProfiler`: Memory usage analysis
- `CPUProfiler`: CPU utilization tracking
- `FunctionProfiler`: Function-level performance analysis

## Integration Patterns

### Unified Error Handling

Consistent error handling across all modules using standardized patterns.

```python
from active_inference.utils.advanced_validation import (
    ValidationError, handle_errors, robust_execution
)
from active_inference.utils.logging_config import get_logger

logger = get_logger(__name__)

class RobustAgent:
    """Example of robust agent with comprehensive error handling."""

    @robust_execution(max_retries=3, fallback_value={'status': 'error'})
    def act(self, observation):
        """Robust action selection with comprehensive error handling."""

        # Input validation
        try:
            validate_array(observation, "observation", expected_shape=(self.obs_dim,))
        except ValidationError as e:
            logger.error(f"Observation validation failed: {e}")
            raise

        # Core processing with monitoring
        with self.performance_monitor.measure("inference"):
            beliefs = self.inference.update_beliefs(self.beliefs, observation)

        with self.performance_monitor.measure("planning"):
            action = self.planning.plan_action(beliefs, self.model)

        # Security validation
        security_result = self.security_validator.validate_secure(
            action, "action", context={"agent_id": self.agent_id}
        )

        if not security_result.is_valid:
            logger.warning(f"Action security validation failed: {security_result.errors}")
            action = self.fallback_action()  # Safe fallback

        return {'status': 'success', 'action': action}

    @handle_errors((InferenceError, PlanningError), log_errors=True)
    def update_model(self, observation, action, reward):
        """Robust model updating with error handling."""

        # Validate inputs
        validate_array(observation, "observation")
        validate_array(action, "action")

        # Update model with caching
        cache_key = f"model_update_{hash(str(observation) + str(action))}"
        cached_update = self.cache.get(cache_key)

        if cached_update is None:
            # Perform expensive update
            update_result = self.model.update(observation, action, reward)
            self.cache.put(cache_key, update_result)
        else:
            update_result = cached_update

        return update_result
```

### Configuration-Driven Behavior

Utilities that adapt behavior based on configuration settings.

```python
from active_inference.utils.config import load_config
from active_inference.utils.logging_config import setup_global_logging
from active_inference.performance.caching import create_cache_from_config

class ConfigurableAgent:
    """Agent that adapts behavior based on configuration."""

    def __init__(self, config_prefix="ACTIVE_INFERENCE_"):
        # Load configuration
        self.config = load_config(prefix=config_prefix)

        # Setup logging based on config
        log_level = get_config_value(self.config, "log_level", str, default="INFO")
        log_file = get_config_value(self.config, "log_file", str, default=None)

        setup_global_logging(
            level=log_level,
            log_file=log_file,
            enable_json=get_config_value(self.config, "json_logging", bool, default=True)
        )

        # Setup caching based on config
        cache_config = {
            'max_size': get_config_value(self.config, "cache_size", int, default=1000),
            'ttl': get_config_value(self.config, "cache_ttl", int, default=300),
            'strategy': get_config_value(self.config, "cache_strategy", str, default="lru")
        }

        self.cache = create_cache_from_config(cache_config)

        # Setup performance monitoring if enabled
        if get_config_value(self.config, "enable_monitoring", bool, default=True):
            self.monitor = PerformanceMonitor(self.__class__.__name__)
        else:
            self.monitor = None

        # Configure security level
        security_level = get_config_value(self.config, "security_level", str, default="standard")
        self.validator = AdvancedValidator(security_level=security_level)

        logger.info(f"Agent configured with security level: {security_level}")
```

### Health-Aware Operations

Operations that adapt based on system health and resource availability.

```python
from active_inference.utils.health_check import HealthMonitor
from active_inference.utils.graceful_degradation import GracefulDegradationManager

class HealthAwareAgent:
    """Agent that adapts behavior based on system health."""

    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.degradation_manager = GracefulDegradationManager()

        # Register health checks
        self.health_monitor.register_component(
            "inference_engine",
            self._check_inference_health
        )

        self.health_monitor.register_component(
            "memory_system",
            self._check_memory_health
        )

    def act(self, observation):
        """Health-aware action selection."""

        # Check system health
        health_status = self.health_monitor.get_health_status()

        if not health_status['healthy']:
            logger.warning(f"System health degraded: {health_status['issues']}")

            # Apply graceful degradation
            degradation_level = self.degradation_manager.assess_degradation_level(
                health_status
            )

            self.degradation_manager.apply_degradation(degradation_level)
            logger.info(f"Applied degradation level: {degradation_level}")

        # Adapt behavior based on health
        if health_status['memory_pressure']:
            # Use memory-efficient methods
            action = self.memory_efficient_act(observation)
        elif health_status['cpu_pressure']:
            # Use faster but less accurate methods
            action = self.fast_approximate_act(observation)
        else:
            # Normal operation
            action = self.full_precision_act(observation)

        return action

    def _check_inference_health(self):
        """Check inference engine health."""
        try:
            # Quick health check
            test_obs = np.random.randn(self.obs_dim)
            test_action = self.act(test_obs)
            return {
                'healthy': True,
                'response_time': 0.001,  # Mock response time
                'accuracy': 0.95  # Mock accuracy
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }

    def _check_memory_health(self):
        """Check memory system health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            return {
                'healthy': available_gb > 1.0,  # At least 1GB available
                'available_gb': available_gb,
                'usage_percent': memory.percent
            }
        except ImportError:
            return {'healthy': True, 'note': 'psutil not available'}
```

## Performance Optimization

### Memory-Efficient Operations

Utilities for memory-efficient computations in resource-constrained environments.

```python
from active_inference.utils.memory_optimization import (
    MemoryEfficientProcessor, StreamingProcessor
)

class MemoryOptimizedAgent:
    """Agent optimized for memory-constrained environments."""

    def __init__(self):
        self.memory_processor = MemoryEfficientProcessor(max_memory_mb=100)
        self.streaming_processor = StreamingProcessor(batch_size=10)

    def process_large_dataset(self, dataset):
        """Process large datasets with memory constraints."""

        # Use streaming processing for large datasets
        results = []
        for batch in self.streaming_processor.stream_batches(dataset):
            batch_results = self.process_batch_memory_efficient(batch)
            results.extend(batch_results)

            # Check memory usage
            memory_usage = self.memory_processor.get_memory_usage()
            if memory_usage > 80:  # 80% of limit
                self.memory_processor.force_garbage_collection()

        return results

    def process_batch_memory_efficient(self, batch):
        """Process batch with memory efficiency."""

        with self.memory_processor.memory_limit(batch_size=len(batch)):
            # Use memory-mapped operations
            beliefs = self.memory_processor.memory_efficient_inference(batch)

            # Clean up intermediate results
            self.memory_processor.cleanup_intermediates()

            return beliefs
```

### Concurrent Processing Utilities

Thread-safe utilities for concurrent active inference operations.

```python
from active_inference.utils.concurrent import (
    ThreadPoolExecutor, ConcurrentProcessor, LockManager
)

class ConcurrentAgent:
    """Agent supporting concurrent operations."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock_manager = LockManager()
        self.concurrent_processor = ConcurrentProcessor()

    def parallel_inference(self, observations_batch):
        """Perform inference on multiple observations concurrently."""

        def single_inference(obs):
            with self.lock_manager.lock('inference'):
                return self.inference.update_beliefs(self.beliefs, obs, self.model)

        # Execute concurrently
        futures = [
            self.executor.submit(single_inference, obs)
            for obs in observations_batch
        ]

        results = [future.result() for future in futures]
        return results

    def concurrent_planning(self, belief_states):
        """Plan actions for multiple belief states concurrently."""

        def single_planning(beliefs):
            with self.lock_manager.lock('planning'):
                return self.planning.plan_action(beliefs, self.model)

        # Use concurrent processor
        results = self.concurrent_processor.process_batch(
            belief_states, single_planning, max_concurrent=4
        )

        return results
```

## Testing and Validation

### Utility Testing Framework

Comprehensive testing utilities for utility functions.

```python
from active_inference.utils.testing import (
    UtilityTester, PerformanceTester, IntegrationTester
)

class UtilsTestSuite:
    """Comprehensive testing for utility functions."""

    def __init__(self):
        self.utility_tester = UtilityTester()
        self.performance_tester = PerformanceTester()
        self.integration_tester = IntegrationTester()

    def run_full_test_suite(self):
        """Run complete utility test suite."""

        test_results = {}

        # Test validation utilities
        test_results['validation'] = self.utility_tester.test_validation_functions([
            'validate_array', 'validate_inputs', 'safe_divide'
        ])

        # Test caching utilities
        test_results['caching'] = self.utility_tester.test_caching_functions([
            'LRUCache', 'AdaptiveCache', 'CacheManager'
        ])

        # Test logging utilities
        test_results['logging'] = self.utility_tester.test_logging_functions([
            'StructuredLogger', 'PerformanceMonitor', 'TelemetryLogger'
        ])

        # Performance testing
        test_results['performance'] = self.performance_tester.test_performance([
            'cache_operations', 'validation_speed', 'logging_overhead'
        ])

        # Integration testing
        test_results['integration'] = self.integration_tester.test_integration([
            'logging_with_validation', 'caching_with_monitoring',
            'concurrent_processing'
        ])

        return test_results

    def benchmark_utilities(self):
        """Benchmark utility function performance."""

        benchmarks = {
            'cache_performance': self._benchmark_caching(),
            'validation_performance': self._benchmark_validation(),
            'logging_performance': self._benchmark_logging()
        }

        return benchmarks
```

## Future Enhancements

### Advanced Utilities

- **Distributed Utilities**: Cross-node utility coordination
- **Machine Learning Utilities**: ML-specific helper functions
- **Real-time Utilities**: High-frequency operation support
- **Edge Computing Utilities**: Resource-constrained device support

### Performance Optimizations

- **GPU Utilities**: GPU-accelerated utility functions
- **Vectorized Operations**: SIMD-optimized computations
- **Memory-Mapped Operations**: Large dataset handling
- **Async Utilities**: Asynchronous operation support

### Security Enhancements

- **Cryptographic Utilities**: Encryption and signing helpers
- **Privacy Utilities**: Differential privacy support
- **Audit Utilities**: Enhanced security event tracking
- **Compliance Utilities**: Regulatory compliance helpers
