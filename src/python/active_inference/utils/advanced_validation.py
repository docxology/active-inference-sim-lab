"""Advanced Validation Framework for Robust Active Inference Systems.

This module implements comprehensive validation, error handling, and monitoring
capabilities for production-grade Active Inference deployments:

- Multi-level input validation with semantic checking
- Real-time health monitoring and anomaly detection
- Graceful error recovery and fallback mechanisms
- Performance monitoring and resource management
- Security validation and threat detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Type
from .logging_config import get_unified_logger
from dataclasses import dataclass, field
import time
import threading
from queue import Queue, Empty
from abc import ABC, abstractmethod
import psutil
import hashlib
import json
import re
from pathlib import Path
from functools import wraps
import traceback
from collections import deque, defaultdict
import warnings
from contextlib import contextmanager


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0
    confidence_score: float = 1.0


@dataclass
class HealthMetrics:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    error_rate: float
    response_time: float
    throughput: float
    timestamp: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security-related event."""
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    source: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationException(Exception):
    """Base exception for validation errors."""
    def __init__(self, message: str, validation_result: ValidationResult = None):
        super().__init__(message)
        self.validation_result = validation_result


class SecurityException(Exception):
    """Exception for security-related issues."""
    def __init__(self, message: str, security_event: SecurityEvent = None):
        super().__init__(message)
        self.security_event = security_event


class PerformanceException(Exception):
    """Exception for performance-related issues."""
    pass


class AdvancedValidator:
    """Advanced validation system with multi-level checks."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.logger = get_unified_logger()
        self.validation_history = deque(maxlen=1000)
        self.error_patterns = defaultdict(int)
        
        # Validation rules
        self.rules = {
            'array_validation': self._validate_array_advanced,
            'model_validation': self._validate_model_parameters,
            'agent_validation': self._validate_agent_state,
            'environment_validation': self._validate_environment_compatibility,
            'performance_validation': self._validate_performance_requirements
        }
    
    def validate_comprehensive(self, 
                             data: Any, 
                             validation_type: str,
                             context: Dict[str, Any] = None) -> ValidationResult:
        """Perform comprehensive validation with multiple checks."""
        start_time = time.time()
        context = context or {}
        
        result = ValidationResult(is_valid=True)
        
        try:
            # Type-specific validation
            if validation_type in self.rules:
                type_result = self.rules[validation_type](data, context)
                self._merge_validation_results(result, type_result)
            
            # Semantic validation
            semantic_result = self._validate_semantic_consistency(data, validation_type, context)
            self._merge_validation_results(result, semantic_result)
            
            # Security validation
            security_result = self._validate_security_constraints(data, context)
            self._merge_validation_results(result, security_result)
            
            # Performance validation
            perf_result = self._validate_performance_impact(data, context)
            self._merge_validation_results(result, perf_result)
            
            # Compute overall confidence
            result.confidence_score = self._compute_validation_confidence(result)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation system error: {str(e)}")
            result.confidence_score = 0.0
            self.logger.log_error(f"Validation system error: {e}")
        
        result.validation_time = time.time() - start_time
        
        # Record validation history
        self.validation_history.append({
            'timestamp': time.time(),
            'validation_type': validation_type,
            'is_valid': result.is_valid,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings),
            'validation_time': result.validation_time
        })
        
        # Update error patterns
        for error in result.errors:
            self.error_patterns[error[:50]] += 1  # First 50 chars
        
        # Raise exception in strict mode
        if self.strict_mode and not result.is_valid:
            raise ValidationException(f"Validation failed: {result.errors}", result)
        
        return result
    
    def _validate_array_advanced(self, array: np.ndarray, context: Dict) -> ValidationResult:
        """Advanced array validation with statistical checks."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Basic array checks
            if not isinstance(array, np.ndarray):
                result.is_valid = False
                result.errors.append(f"Expected numpy array, got {type(array)}")
                return result
            
            # Shape validation
            expected_shape = context.get('expected_shape')
            if expected_shape and array.shape != expected_shape:
                result.is_valid = False
                result.errors.append(f"Shape mismatch: expected {expected_shape}, got {array.shape}")
            
            # Data type validation
            expected_dtype = context.get('expected_dtype')
            if expected_dtype and array.dtype != expected_dtype:
                result.warnings.append(f"Dtype mismatch: expected {expected_dtype}, got {array.dtype}")
            
            # Value range validation
            min_val = context.get('min_value')
            max_val = context.get('max_value')
            
            if min_val is not None and np.any(array < min_val):
                result.errors.append(f"Values below minimum {min_val}: {np.sum(array < min_val)} elements")
                result.is_valid = False
            
            if max_val is not None and np.any(array > max_val):
                result.errors.append(f"Values above maximum {max_val}: {np.sum(array > max_val)} elements")
                result.is_valid = False
            
            # Statistical validation
            if array.size > 0:
                # Check for infinite or NaN values
                if not np.isfinite(array).all():
                    nan_count = np.isnan(array).sum()
                    inf_count = np.isinf(array).sum()
                    result.errors.append(f"Non-finite values: {nan_count} NaN, {inf_count} infinite")
                    result.is_valid = False
                
                # Statistical anomaly detection
                if array.size > 3:
                    mean = np.mean(array)
                    std = np.std(array)
                    
                    # Z-score outlier detection
                    if std > 0:
                        z_scores = np.abs((array - mean) / std)
                        outliers = np.sum(z_scores > 3)
                        if outliers > array.size * 0.1:  # More than 10% outliers
                            result.warnings.append(f"High outlier rate: {outliers}/{array.size} ({outliers/array.size:.1%})")
                    
                    # Distribution checks
                    skewness = self._compute_skewness(array)
                    if abs(skewness) > 2:
                        result.warnings.append(f"High skewness: {skewness:.2f} (may indicate data issues)")
                    
                    # Entropy check for information content
                    if array.size > 10:
                        entropy = self._compute_entropy(array)
                        result.metadata['entropy'] = entropy
                        if entropy < 0.1:
                            result.warnings.append(f"Low entropy: {entropy:.3f} (data may be too uniform)")
            
            # Memory usage validation
            memory_mb = array.nbytes / (1024 * 1024)
            max_memory = context.get('max_memory_mb', 1000)  # 1GB default
            if memory_mb > max_memory:
                result.warnings.append(f"Large memory usage: {memory_mb:.1f}MB > {max_memory}MB")
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Array validation error: {str(e)}")
        
        return result
    
    def _validate_model_parameters(self, params: Dict, context: Dict) -> ValidationResult:
        """Validate model parameters for consistency and safety."""
        result = ValidationResult(is_valid=True)
        
        try:
            if not isinstance(params, dict):
                result.is_valid = False
                result.errors.append(f"Expected dictionary, got {type(params)}")
                return result
            
            # Check required parameters
            required_params = context.get('required_params', [])
            for param in required_params:
                if param not in params:
                    result.errors.append(f"Missing required parameter: {param}")
                    result.is_valid = False
            
            # Validate parameter ranges
            param_ranges = context.get('param_ranges', {})
            for param, value in params.items():
                if param in param_ranges:
                    min_val, max_val = param_ranges[param]
                    if not (min_val <= value <= max_val):
                        result.errors.append(f"Parameter {param}={value} outside range [{min_val}, {max_val}]")
                        result.is_valid = False
            
            # Check for parameter interactions
            interactions = context.get('parameter_interactions', [])
            for interaction in interactions:
                if not self._check_parameter_interaction(params, interaction):
                    result.warnings.append(f"Parameter interaction warning: {interaction['description']}")
            
            # Validate parameter stability
            if 'learning_rate' in params and 'temperature' in params:
                lr = params['learning_rate']
                temp = params['temperature']
                
                # High learning rate + low temperature can cause instability
                if lr > 0.1 and temp < 0.1:
                    result.warnings.append("High learning rate with low temperature may cause instability")
                
                # Very low learning rate may prevent learning
                if lr < 1e-6:
                    result.warnings.append(f"Very low learning rate ({lr}) may prevent effective learning")
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Model parameter validation error: {str(e)}")
        
        return result
    
    def _validate_agent_state(self, agent_state: Dict, context: Dict) -> ValidationResult:
        """Validate agent state for consistency and health."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check agent health indicators
            error_rate = agent_state.get('error_rate', 0)
            if error_rate > 0.1:  # More than 10% errors
                result.warnings.append(f"High agent error rate: {error_rate:.1%}")
            
            # Check belief consistency
            beliefs = agent_state.get('beliefs', {})
            if beliefs:
                for belief_name, belief_data in beliefs.items():
                    if 'variance' in belief_data:
                        variance = belief_data['variance']
                        if np.any(variance < 0):
                            result.errors.append(f"Negative variance in belief '{belief_name}'")
                            result.is_valid = False
                        
                        # Check for collapsed beliefs (very low variance)
                        if np.any(variance < 1e-8):
                            result.warnings.append(f"Very low variance in belief '{belief_name}' (may indicate collapse)")
            
            # Check temporal consistency
            step_count = agent_state.get('step_count', 0)
            episode_count = agent_state.get('episode_count', 0)
            if step_count > 0 and episode_count == 0:
                result.warnings.append("Steps taken but no episodes recorded")
            
            # Check learning progress
            total_reward = agent_state.get('total_reward', 0)
            if step_count > 1000 and total_reward == 0:
                result.warnings.append("No reward accumulated after many steps (learning may be ineffective)")
            
            # Memory usage check
            history_length = agent_state.get('history_length', 0)
            max_history = context.get('max_history_length', 10000)
            if history_length > max_history * 0.9:
                result.warnings.append(f"History length {history_length} approaching maximum {max_history}")
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Agent state validation error: {str(e)}")
        
        return result
    
    def _validate_environment_compatibility(self, env_info: Dict, context: Dict) -> ValidationResult:
        """Validate environment compatibility and configuration."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check dimensions compatibility
            obs_dim = env_info.get('observation_dim')
            action_dim = env_info.get('action_dim')
            agent_obs_dim = context.get('agent_obs_dim')
            agent_action_dim = context.get('agent_action_dim')
            
            if obs_dim and agent_obs_dim and obs_dim != agent_obs_dim:
                result.errors.append(f"Observation dimension mismatch: env={obs_dim}, agent={agent_obs_dim}")
                result.is_valid = False
            
            if action_dim and agent_action_dim and action_dim != agent_action_dim:
                result.errors.append(f"Action dimension mismatch: env={action_dim}, agent={agent_action_dim}")
                result.is_valid = False
            
            # Check environment stability
            episode_length = env_info.get('episode_length', 0)
            max_episode_length = context.get('max_episode_length', 10000)
            if episode_length > max_episode_length:
                result.warnings.append(f"Very long episodes ({episode_length}) may cause memory issues")
            
            # Check reward range
            reward_range = env_info.get('reward_range')
            if reward_range:
                min_reward, max_reward = reward_range
                if max_reward - min_reward > 1000:
                    result.warnings.append(f"Wide reward range [{min_reward}, {max_reward}] may cause learning instability")
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Environment compatibility validation error: {str(e)}")
        
        return result
    
    def _validate_performance_requirements(self, perf_data: Dict, context: Dict) -> ValidationResult:
        """Validate performance requirements and constraints."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Response time validation
            response_time = perf_data.get('response_time', 0)
            max_response_time = context.get('max_response_time', 1.0)  # 1 second default
            if response_time > max_response_time:
                result.warnings.append(f"Slow response time: {response_time:.3f}s > {max_response_time}s")
            
            # Throughput validation
            throughput = perf_data.get('throughput', 0)
            min_throughput = context.get('min_throughput', 1.0)  # 1 ops/sec default
            if throughput < min_throughput:
                result.warnings.append(f"Low throughput: {throughput:.2f} < {min_throughput} ops/sec")
            
            # Memory usage validation
            memory_usage = perf_data.get('memory_usage_mb', 0)
            max_memory = context.get('max_memory_mb', 2000)  # 2GB default
            if memory_usage > max_memory:
                result.errors.append(f"Memory usage {memory_usage:.1f}MB exceeds limit {max_memory}MB")
                result.is_valid = False
            
            # CPU usage validation
            cpu_usage = perf_data.get('cpu_usage_percent', 0)
            max_cpu = context.get('max_cpu_percent', 90)  # 90% default
            if cpu_usage > max_cpu:
                result.warnings.append(f"High CPU usage: {cpu_usage:.1f}% > {max_cpu}%")
            
            # Error rate validation
            error_rate = perf_data.get('error_rate', 0)
            max_error_rate = context.get('max_error_rate', 0.05)  # 5% default
            if error_rate > max_error_rate:
                result.errors.append(f"High error rate: {error_rate:.1%} > {max_error_rate:.1%}")
                result.is_valid = False
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Performance validation error: {str(e)}")
        
        return result
    
    def _validate_semantic_consistency(self, data: Any, validation_type: str, context: Dict) -> ValidationResult:
        """Validate semantic consistency and logical constraints."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Contextual validation based on type
            if validation_type == 'agent_validation':
                # Check temporal consistency
                if isinstance(data, dict):
                    step_count = data.get('step_count', 0)
                    episode_count = data.get('episode_count', 0)
                    
                    if step_count < episode_count:
                        result.warnings.append("Steps less than episodes (unusual pattern)")
            
            elif validation_type == 'array_validation':
                # Check array semantic consistency
                if isinstance(data, np.ndarray):
                    # For probability distributions, check sum to 1
                    if context.get('is_probability_distribution'):
                        array_sum = np.sum(data)
                        if not np.isclose(array_sum, 1.0, atol=1e-6):
                            result.errors.append(f"Probability distribution sum {array_sum:.6f} != 1.0")
                            result.is_valid = False
                    
                    # For correlation matrices, check symmetry
                    if context.get('is_correlation_matrix'):
                        if data.shape[0] != data.shape[1]:
                            result.errors.append("Correlation matrix must be square")
                            result.is_valid = False
                        elif not np.allclose(data, data.T, atol=1e-6):
                            result.warnings.append("Correlation matrix not symmetric")
        
        except Exception as e:
            result.warnings.append(f"Semantic validation error: {str(e)}")
        
        return result
    
    def _validate_security_constraints(self, data: Any, context: Dict) -> ValidationResult:
        """Validate security constraints and detect potential threats."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check for potential injection attacks in string data
            if isinstance(data, str):
                suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(', '__import__']
                for pattern in suspicious_patterns:
                    if pattern.lower() in data.lower():
                        result.errors.append(f"Potential code injection detected: '{pattern}'")
                        result.is_valid = False
            
            # Check for excessively large data (DoS attack)
            data_size = self._estimate_data_size(data)
            max_size_mb = context.get('max_data_size_mb', 100)  # 100MB default
            if data_size > max_size_mb * 1024 * 1024:
                result.errors.append(f"Data size {data_size/(1024*1024):.1f}MB exceeds security limit {max_size_mb}MB")
                result.is_valid = False
            
            # Check for suspicious file paths
            if isinstance(data, (str, Path)):
                path_str = str(data)
                suspicious_paths = ['../', '..\\', '/etc/', 'C:\\Windows', '/proc/']
                for sus_path in suspicious_paths:
                    if sus_path in path_str:
                        result.warnings.append(f"Suspicious path pattern detected: '{sus_path}'")
            
            # Check for known malicious patterns in arrays
            if isinstance(data, np.ndarray) and data.dtype == object:
                result.warnings.append("Object arrays may contain arbitrary code")
        
        except Exception as e:
            result.warnings.append(f"Security validation error: {str(e)}")
        
        return result
    
    def _validate_performance_impact(self, data: Any, context: Dict) -> ValidationResult:
        """Validate potential performance impact of data."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Estimate computational complexity
            if isinstance(data, np.ndarray):
                # Large arrays may cause performance issues
                if data.size > 1000000:  # 1M elements
                    result.warnings.append(f"Large array ({data.size} elements) may impact performance")
                
                # High-dimensional arrays
                if data.ndim > 5:
                    result.warnings.append(f"High-dimensional array ({data.ndim}D) may be computationally expensive")
            
            elif isinstance(data, dict):
                # Deep nested structures
                max_depth = self._compute_dict_depth(data)
                if max_depth > 10:
                    result.warnings.append(f"Deeply nested structure (depth {max_depth}) may impact performance")
                
                # Large number of keys
                if len(data) > 1000:
                    result.warnings.append(f"Large dictionary ({len(data)} keys) may impact performance")
        
        except Exception as e:
            result.warnings.append(f"Performance impact validation error: {str(e)}")
        
        return result
    
    def _merge_validation_results(self, target: ValidationResult, source: ValidationResult):
        """Merge validation results."""
        target.errors.extend(source.errors)
        target.warnings.extend(source.warnings)
        target.metadata.update(source.metadata)
        target.is_valid = target.is_valid and source.is_valid
    
    def _compute_validation_confidence(self, result: ValidationResult) -> float:
        """Compute confidence score for validation result."""
        if not result.is_valid:
            return 0.0
        
        # Base confidence
        confidence = 1.0
        
        # Reduce confidence based on warnings
        warning_penalty = min(0.5, len(result.warnings) * 0.1)
        confidence -= warning_penalty
        
        # Validation time penalty (slower validation = less confidence)
        if result.validation_time > 1.0:
            time_penalty = min(0.2, (result.validation_time - 1.0) * 0.05)
            confidence -= time_penalty
        
        return max(0.0, confidence)
    
    def _check_parameter_interaction(self, params: Dict, interaction: Dict) -> bool:
        """Check parameter interaction constraint."""
        try:
            # Example: check if learning_rate * temperature < threshold
            condition = interaction.get('condition')
            if condition == 'lr_temp_product':
                lr = params.get('learning_rate', 0)
                temp = params.get('temperature', 1)
                threshold = interaction.get('threshold', 0.1)
                return lr * temp <= threshold
            
            # Add more interaction checks as needed
            return True
        
        except Exception:
            return True  # Assume valid if check fails
    
    def _compute_skewness(self, array: np.ndarray) -> float:
        """Compute skewness of array."""
        if array.size < 3:
            return 0.0
        
        mean = np.mean(array)
        std = np.std(array)
        
        if std == 0:
            return 0.0
        
        return np.mean(((array - mean) / std) ** 3)
    
    def _compute_entropy(self, array: np.ndarray) -> float:
        """Compute entropy of array values."""
        try:
            # Discretize continuous values
            hist, _ = np.histogram(array, bins=min(50, array.size // 10))
            hist = hist[hist > 0]  # Remove zero bins
            
            # Normalize to probabilities
            probs = hist / np.sum(hist)
            
            # Compute entropy
            entropy = -np.sum(probs * np.log2(probs))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(probs))
            return entropy / max_entropy if max_entropy > 0 else 0.0
        
        except Exception:
            return 0.5  # Default entropy
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate memory size of data object."""
        try:
            if isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_data_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._estimate_data_size(k) + self._estimate_data_size(v) 
                          for k, v in data.items())
            else:
                return 64  # Rough estimate for other objects
        except Exception:
            return 0
    
    def _compute_dict_depth(self, data: Dict, current_depth: int = 0) -> int:
        """Compute maximum depth of nested dictionary."""
        if not isinstance(data, dict):
            return current_depth
        
        max_depth = current_depth
        for value in data.values():
            if isinstance(value, dict):
                depth = self._compute_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation system statistics."""
        if not self.validation_history:
            return {'status': 'no_validations_performed'}
        
        recent_validations = list(self.validation_history)[-100:]  # Last 100
        
        stats = {
            'total_validations': len(self.validation_history),
            'recent_validations': len(recent_validations),
            'success_rate': np.mean([v['is_valid'] for v in recent_validations]),
            'avg_validation_time': np.mean([v['validation_time'] for v in recent_validations]),
            'avg_errors_per_validation': np.mean([v['error_count'] for v in recent_validations]),
            'avg_warnings_per_validation': np.mean([v['warning_count'] for v in recent_validations]),
            'most_common_errors': dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
            'validation_types': list(set(v['validation_type'] for v in recent_validations))
        }
        
        return stats


class HealthMonitor:
    """Real-time health monitoring system."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.logger = get_unified_logger()
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)
        self.alert_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'error_rate': 0.1,
            'response_time': 2.0
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.error_count = 0
        self.total_requests = 0
        self.last_alert_time = 0
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.log_info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.logger.log_info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Check thresholds and trigger alerts
                self._check_alert_thresholds(metrics)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.log_error(f"Health monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> HealthMetrics:
        """Collect current system health metrics."""
        try:
            # System resource usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            # Application metrics
            error_rate = self.error_count / max(1, self.total_requests)
            avg_response_time = np.mean(self.request_times) if self.request_times else 0
            throughput = len(self.request_times) / max(1, self.monitoring_interval)
            
            return HealthMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_info.percent,
                disk_usage=disk_info.percent,
                error_rate=error_rate,
                response_time=avg_response_time,
                throughput=throughput,
                timestamp=time.time()
            )
        
        except Exception as e:
            self.logger.log_error(f"Error collecting system metrics: {e}", component="advanced_validation")
            return HealthMetrics(
                cpu_usage=0, memory_usage=0, disk_usage=0,
                error_rate=0, response_time=0, throughput=0,
                timestamp=time.time()
            )
    
    def _check_alert_thresholds(self, metrics: HealthMetrics):
        """Check if any metrics exceed alert thresholds."""
        current_time = time.time()
        
        # Prevent alert spam (minimum 60 seconds between similar alerts)
        if current_time - self.last_alert_time < 60:
            return
        
        alerts = []
        
        # Check each threshold
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.disk_usage > self.alert_thresholds['disk_usage']:
            alerts.append(f"High disk usage: {metrics.disk_usage:.1f}%")
        
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics.error_rate:.1%}")
        
        if metrics.response_time > self.alert_thresholds['response_time']:
            alerts.append(f"Slow response time: {metrics.response_time:.3f}s")
        
        # Trigger alerts
        if alerts:
            self.last_alert_time = current_time
            for alert_msg in alerts:
                self.logger.log_warning(f"Health alert: {alert_msg}", component="advanced_validation")
                for callback in self.alert_callbacks:
                    try:
                        callback(alert_msg, metrics)
                    except Exception as e:
                        self.logger.log_error(f"Alert callback error: {e}", component="advanced_validation")
    
    def add_alert_callback(self, callback: Callable[[str, HealthMetrics], None]):
        """Add callback function for health alerts."""
        self.alert_callbacks.append(callback)
    
    def record_request_time(self, request_time: float):
        """Record response time for a request."""
        self.request_times.append(request_time)
        self.total_requests += 1
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
    
    def get_current_health(self) -> HealthMetrics:
        """Get current health metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        else:
            return self._collect_system_metrics()
    
    def get_health_history(self, last_n: int = 100) -> List[HealthMetrics]:
        """Get recent health metrics history."""
        return list(self.metrics_history)[-last_n:]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 readings
        
        summary = {
            'monitoring_active': self.is_monitoring,
            'total_metrics_collected': len(self.metrics_history),
            'recent_metrics_count': len(recent_metrics),
            'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_disk_usage': np.mean([m.disk_usage for m in recent_metrics]),
            'avg_error_rate': np.mean([m.error_rate for m in recent_metrics]),
            'avg_response_time': np.mean([m.response_time for m in recent_metrics]),
            'avg_throughput': np.mean([m.throughput for m in recent_metrics]),
            'alert_thresholds': self.alert_thresholds,
            'total_requests': self.total_requests,
            'total_errors': self.error_count,
            'overall_error_rate': self.error_count / max(1, self.total_requests)
        }
        
        return summary


class SecurityMonitor:
    """Security monitoring and threat detection system."""
    
    def __init__(self):
        self.logger = get_unified_logger()
        self.security_events = deque(maxlen=10000)
        self.threat_patterns = self._initialize_threat_patterns()
        self.blocked_ips = set()
        self.rate_limits = defaultdict(deque)  # IP -> request times
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_minute = 100
    
    def _initialize_threat_patterns(self) -> Dict[str, Dict]:
        """Initialize threat detection patterns."""
        return {
            'sql_injection': {
                'patterns': ['union select', 'drop table', "'; --", "' or 1=1"],
                'severity': 'high'
            },
            'code_injection': {
                'patterns': ['eval(', 'exec(', '__import__', 'subprocess.'],
                'severity': 'critical'
            },
            'path_traversal': {
                'patterns': ['../', '..\\', '/etc/passwd', 'C:\\Windows'],
                'severity': 'high'
            },
            'xss_attempt': {
                'patterns': ['<script', 'javascript:', 'onerror=', 'onload='],
                'severity': 'medium'
            }
        }
    
    def validate_input_security(self, input_data: str, source_ip: str = None) -> ValidationResult:
        """Validate input data for security threats."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check rate limiting
            if source_ip and not self._check_rate_limit(source_ip):
                result.is_valid = False
                result.errors.append(f"Rate limit exceeded for IP {source_ip}")
                self._record_security_event("rate_limit_exceeded", "medium", 
                                           f"IP {source_ip} exceeded rate limit", source_ip or "unknown")
                return result
            
            # Check against threat patterns
            input_lower = input_data.lower()
            
            for threat_type, threat_info in self.threat_patterns.items():
                for pattern in threat_info['patterns']:
                    if pattern in input_lower:
                        severity = threat_info['severity']
                        result.is_valid = False
                        result.errors.append(f"{threat_type.replace('_', ' ').title()} detected: '{pattern}'")
                        
                        # Record security event
                        self._record_security_event(
                            threat_type, severity,
                            f"Threat pattern '{pattern}' detected in input",
                            source_ip or "unknown",
                            {'input_sample': input_data[:100]}  # First 100 chars
                        )
                        
                        # Block IP for critical threats
                        if severity == 'critical' and source_ip:
                            self._block_ip(source_ip)
            
            # Check input length (potential DoS)
            if len(input_data) > 1000000:  # 1MB
                result.warnings.append(f"Very large input: {len(input_data)} bytes")
                self._record_security_event(
                    "large_input", "low",
                    f"Large input received: {len(input_data)} bytes",
                    source_ip or "unknown"
                )
        
        except Exception as e:
            self.logger.log_error(f"Security validation error: {e}")
            result.warnings.append(f"Security validation error: {str(e, component="advanced_validation")}", component="advanced_validation")
        
        return result
    
    def _check_rate_limit(self, source_ip: str) -> bool:
        """Check if IP is within rate limits."""
        current_time = time.time()
        
        # Clean old entries
        self.rate_limits[source_ip] = deque(
            [t for t in self.rate_limits[source_ip] 
             if current_time - t <= self.rate_limit_window],
            maxlen=self.max_requests_per_minute
        )
        
        # Check current count
        if len(self.rate_limits[source_ip]) >= self.max_requests_per_minute:
            return False
        
        # Record this request
        self.rate_limits[source_ip].append(current_time)
        return True
    
    def _block_ip(self, source_ip: str):
        """Block an IP address."""
        self.blocked_ips.add(source_ip)
        self.logger.log_warning(f"IP {source_ip} has been blocked due to security threat", component="advanced_validation")
        
        # Record blocking event
        self._record_security_event(
            "ip_blocked", "high",
            f"IP {source_ip} blocked due to security threat",
            source_ip
        )
    
    def _record_security_event(self, event_type: str, severity: str, 
                              description: str, source: str, metadata: Dict = None):
        """Record a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            source=source,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Log based on severity
        log_msg = f"Security event [{severity}]: {description} from {source}"
        if severity == 'critical':
            self.logger.critical(log_msg)
        elif severity == 'high':
            self.logger.log_error(log_msg)
        elif severity == 'medium':
            self.logger.log_warning(log_msg, component="advanced_validation")
        else:
            self.logger.log_info(log_msg, component="advanced_validation")
    
    def is_ip_blocked(self, source_ip: str) -> bool:
        """Check if an IP is blocked."""
        return source_ip in self.blocked_ips
    
    def unblock_ip(self, source_ip: str):
        """Unblock an IP address."""
        if source_ip in self.blocked_ips:
            self.blocked_ips.remove(source_ip)
            self.logger.log_info(f"IP {source_ip} has been unblocked", component="advanced_validation")
    
    def get_security_summary(self, component="advanced_validation") -> Dict[str, Any]:
        """Get security monitoring summary."""
        recent_events = [e for e in self.security_events 
                        if time.time() - e.timestamp <= 3600]  # Last hour
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
        
        return {
            'total_security_events': len(self.security_events),
            'recent_events_1h': len(recent_events),
            'blocked_ips_count': len(self.blocked_ips),
            'blocked_ips': list(self.blocked_ips),
            'recent_event_types': dict(event_counts),
            'recent_severity_distribution': dict(severity_counts),
            'threat_patterns_monitored': len(self.threat_patterns),
            'rate_limit_active_ips': len([ip for ip, times in self.rate_limits.items() if times])
        }


def robust_execution(max_retries: int = 3, 
                    fallback_value: Any = None,
                    exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Decorator for robust execution with retries and fallback."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Exponential backoff
                        sleep_time = 0.1 * (2 ** attempt)
                        time.sleep(sleep_time)
                        
                        get_unified_logger().warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying..."
                        )
                    else:
                        get_unified_logger().error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
            
            # Return fallback value if all attempts failed
            if fallback_value is not None:
                get_unified_logger().info(
                    f"Using fallback value for {func.__name__}"
                )
                return fallback_value
            
            # Re-raise the last exception if no fallback
            raise last_exception
        
        return wrapper
    return decorator


@contextmanager
def error_recovery_context(recovery_actions: List[Callable] = None):
    """Context manager for automatic error recovery."""
    recovery_actions = recovery_actions or []
    
    try:
        yield
    except Exception as e:
        get_unified_logger().error(f"Error occurred: {e}")
        
        # Execute recovery actions
        for action in recovery_actions:
            try:
                action()
                get_unified_logger().info(f"Recovery action executed: {action.__name__}")
            except Exception as recovery_error:
                get_unified_logger().error(
                    f"Recovery action failed: {recovery_error}"
                )
        
        # Re-raise the original exception
        raise


class RobustActiveInferenceFramework:
    """Robust Active Inference framework with comprehensive error handling."""
    
    def __init__(self):
        self.logger = get_unified_logger()
        
        # Initialize monitoring and validation systems
        self.validator = AdvancedValidator(strict_mode=False)
        self.health_monitor = HealthMonitor(monitoring_interval=10.0)
        self.security_monitor = SecurityMonitor()
        
        # Framework state
        self.is_initialized = False
        self.framework_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'start_time': time.time()
        }
        
        # Start monitoring
        self.health_monitor.start_monitoring()
        
        self.logger.log_info("Robust Active Inference Framework initialized", component="advanced_validation")
    
    @robust_execution(max_retries=2, fallback_value={'status': 'error', 'value': None})
    def safe_agent_operation(self, agent: Any, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Safely execute agent operation with comprehensive error handling."""
        operation_start_time = time.time()
        
        try:
            self.framework_stats['total_operations'] += 1
            
            # Validate agent state before operation
            if hasattr(agent, 'get_statistics'):
                agent_state = agent.get_statistics()
                validation_result = self.validator.validate_comprehensive(
                    agent_state, 'agent_validation'
                )
                
                if not validation_result.is_valid:
                    raise ValidationException(f"Agent validation failed: {validation_result.errors}")
            
            # Execute the operation
            if hasattr(agent, operation):
                method = getattr(agent, operation)
                result = method(*args, **kwargs)
            else:
                raise AttributeError(f"Agent does not have method '{operation}'")
            
            # Record successful operation
            operation_time = time.time() - operation_start_time
            self.health_monitor.record_request_time(operation_time)
            self.framework_stats['successful_operations'] += 1
            
            return {'status': 'success', 'value': result, 'operation_time': operation_time}
        
        except Exception as e:
            # Record failed operation
            self.health_monitor.record_error()
            self.framework_stats['failed_operations'] += 1
            
            # Log error with context
            self.logger.log_error(f"Agent operation '{operation}' failed: {e}")
            self.logger.log_debug(f"Traceback: {traceback.format_exc()}", component="advanced_validation")
            
            # Return error information
            return {
                'status': 'error',
                'error': str(e),
                'operation': operation,
                'operation_time': time.time() - operation_start_time
            }
    
    def get_framework_health(self) -> Dict[str, Any]:
        """Get comprehensive framework health status."""
        health_summary = self.health_monitor.get_health_summary()
        validation_stats = self.validator.get_validation_statistics()
        security_summary = self.security_monitor.get_security_summary()
        
        return {
            'framework_stats': self.framework_stats,
            'health_monitoring': health_summary,
            'validation_stats': validation_stats,
            'security_summary': security_summary,
            'uptime': time.time() - self.framework_stats['start_time'],
            'overall_health_score': self._compute_overall_health_score(health_summary, validation_stats, security_summary)
        }
    
    def _compute_overall_health_score(self, health: Dict, validation: Dict, security: Dict) -> float:
        """Compute overall health score (0-1)."""
        try:
            # Health component (40%)
            health_score = 1.0
            if 'avg_cpu_usage' in health:
                health_score -= min(0.5, health['avg_cpu_usage'] / 200)  # Penalty for high CPU
            if 'overall_error_rate' in health:
                health_score -= min(0.3, health['overall_error_rate'] * 3)  # Penalty for errors
            health_component = max(0, health_score) * 0.4
            
            # Validation component (30%)
            validation_score = 1.0
            if 'success_rate' in validation:
                validation_score = validation['success_rate']
            validation_component = validation_score * 0.3
            
            # Security component (30%)
            security_score = 1.0
            if 'recent_events_1h' in security:
                # Penalty for recent security events
                security_score -= min(0.5, security['recent_events_1h'] / 20)
            if 'blocked_ips_count' in security:
                # Small penalty for blocked IPs (indicates threats but also protection)
                security_score -= min(0.1, security['blocked_ips_count'] / 100)
            security_component = max(0, security_score) * 0.3
            
            overall_score = health_component + validation_component + security_component
            return min(1.0, max(0.0, overall_score))
        
        except Exception as e:
            self.logger.log_error(f"Error computing health score: {e}")
            return 0.5  # Default score
    
    def shutdown(self):
        """Gracefully shutdown the framework."""
        self.logger.log_info("Shutting down Robust Active Inference Framework")
        
        try:
            self.health_monitor.stop_monitoring()
            
            # Log final statistics
            final_stats = self.get_framework_health()
            self.logger.log_info(f"Final framework statistics: {final_stats['framework_stats']}")
            
        except Exception as e:
            self.logger.log_error(f"Error during shutdown: {e}")
        
        self.logger.log_info("Framework shutdown complete")


# Backward compatibility: Basic validation functions and classes from utils/validation.py

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ActiveInferenceError(Exception):
    """Base exception for active inference specific errors."""
    pass


class ModelError(ActiveInferenceError):
    """Exception for generative model errors."""
    pass


class InferenceError(ActiveInferenceError):
    """Exception for inference engine errors."""
    pass


class PlanningError(ActiveInferenceError):
    """Exception for planning errors."""
    pass


def validate_array(arr: np.ndarray,
                  name: str,
                  expected_shape: Optional[Tuple] = None,
                  expected_dtype: Optional[np.dtype] = None,
                  min_val: Optional[float] = None,
                  max_val: Optional[float] = None,
                  allow_nan: bool = False,
                  allow_inf: bool = False) -> None:
    """
    Validate numpy array properties.

    Args:
        arr: Array to validate
        name: Name of the array for error messages
        expected_shape: Expected shape (None to skip check)
        expected_dtype: Expected data type (None to skip check)
        min_val: Minimum allowed value (None to skip check)
        max_val: Maximum allowed value (None to skip check)
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether infinite values are allowed

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"{name} must be a numpy array, got {type(arr)}")

    if arr.size == 0:
        raise ValidationError(f"{name} cannot be empty")

    if expected_shape is not None and arr.shape != expected_shape:
        raise ValidationError(f"{name} has shape {arr.shape}, expected {expected_shape}")

    if expected_dtype is not None and arr.dtype != expected_dtype:
        raise ValidationError(f"{name} has dtype {arr.dtype}, expected {expected_dtype}")

    if not allow_nan and np.any(np.isnan(arr)):
        raise ValidationError(f"{name} contains NaN values")

    if not allow_inf and np.any(np.isinf(arr)):
        raise ValidationError(f"{name} contains infinite values")

    if min_val is not None and np.any(arr < min_val):
        raise ValidationError(f"{name} contains values below {min_val}")

    if max_val is not None and np.any(arr > max_val):
        raise ValidationError(f"{name} contains values above {max_val}")


class SecurityValidator:
    """Security validation for Active Inference inputs."""

    def __init__(self):
        self.threat_patterns = {
            'sql_injection': [r'union\s+select', r'drop\s+table', r'insert\s+into'],
            'xss': [r'<script', r'javascript:', r'onload\s*='],
            'command_injection': [r';\s*rm\s+', r';\s*cat\s+', r'&&\s*rm']
        }

    def validate_input(self, input_data: Any) -> bool:
        """Basic security validation of input data."""
        try:
            if isinstance(input_data, np.ndarray):
                # Check for NaN/Inf values
                if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
                    return False
                # Check for reasonable size
                if input_data.nbytes > 10**7:  # 10MB limit
                    return False
            elif isinstance(input_data, str):
                # Check for malicious patterns
                input_lower = input_data.lower()
                for category, patterns in self.threat_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, input_lower):
                            return False
            return True
        except Exception:
            return False


class AdvancedInputValidator:
    """Advanced input validation with detailed checks."""

    def __init__(self):
        self.max_size = 10**6  # 1MB
        self.validation_cache = {}

    def validate_comprehensive(self, input_data: Any) -> Dict[str, bool]:
        """Comprehensive validation returning detailed results."""
        results = {
            'size_valid': True,
            'type_valid': True,
            'content_valid': True,
            'security_valid': True
        }

        try:
            # Size validation
            if hasattr(input_data, 'nbytes'):
                results['size_valid'] = input_data.nbytes <= self.max_size

            # Type validation
            results['type_valid'] = isinstance(input_data, (np.ndarray, list, int, float))

            # Content validation
            if isinstance(input_data, np.ndarray):
                results['content_valid'] = not (np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)))

            return results
        except Exception:
            return {k: False for k in results.keys()}


def validate_matrix(matrix: np.ndarray,
                   name: str,
                   square: bool = False,
                   positive_definite: bool = False,
                   symmetric: bool = False) -> None:
    """
    Validate matrix properties.

    Args:
        matrix: Matrix to validate
        name: Name for error messages
        square: Whether matrix must be square
        positive_definite: Whether matrix must be positive definite
        symmetric: Whether matrix must be symmetric

    Raises:
        ValidationError: If validation fails
    """
    validate_array(matrix, name)

    if matrix.ndim != 2:
        raise ValidationError(f"{name} must be 2D, got {matrix.ndim}D")

    if square and matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(f"{name} must be square, got shape {matrix.shape}")

    if symmetric:
        if not square:
            raise ValidationError(f"{name} must be square to be symmetric")

        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10):
            raise ValidationError(f"{name} is not symmetric")

    if positive_definite:
        if not square:
            raise ValidationError(f"{name} must be square to be positive definite")

        try:
            eigenvals = np.linalg.eigvals(matrix)
            if np.any(eigenvals <= 1e-10):
                raise ValidationError(f"{name} is not positive definite (min eigenvalue: {eigenvals.min()})")
        except np.linalg.LinAlgError:
            raise ValidationError(f"{name} eigenvalue computation failed - likely not positive definite")


def validate_probability_distribution(probs: np.ndarray, name: str) -> None:
    """
    Validate probability distribution.

    Args:
        probs: Probability values
        name: Name for error messages

    Raises:
        ValidationError: If not a valid probability distribution
    """
    validate_array(probs, name, min_val=0.0, max_val=1.0)

    prob_sum = np.sum(probs)
    if not np.isclose(prob_sum, 1.0, rtol=1e-6, atol=1e-6):
        raise ValidationError(f"{name} probabilities sum to {prob_sum}, expected 1.0")


def validate_belief_state(beliefs_dict: Dict[str, Any]) -> None:
    """
    Validate belief state dictionary.

    Args:
        beliefs_dict: Dictionary of belief components

    Raises:
        ValidationError: If belief state is invalid
    """
    if not isinstance(beliefs_dict, dict):
        raise ValidationError(f"Beliefs must be a dictionary, got {type(beliefs_dict)}")

    if len(beliefs_dict) == 0:
        raise ValidationError("Beliefs dictionary cannot be empty")

    for name, belief in beliefs_dict.items():
        if not hasattr(belief, 'mean') or not hasattr(belief, 'variance'):
            raise ValidationError(f"Belief '{name}' must have 'mean' and 'variance' attributes")

        try:
            validate_array(belief.mean, f"belief '{name}' mean")
            validate_array(belief.variance, f"belief '{name}' variance", min_val=0.0)

            # Check matching dimensions
            if belief.mean.shape != belief.variance.shape:
                raise ValidationError(
                    f"Belief '{name}' mean and variance shapes don't match: "
                    f"{belief.mean.shape} vs {belief.variance.shape}"
                )
        except AttributeError as e:
            raise ValidationError(f"Belief '{name}' has invalid structure: {e}")


def validate_dimensions(state_dim: int, obs_dim: int, action_dim: int) -> None:
    """
    Validate dimension parameters.

    Args:
        state_dim: Hidden state dimensionality
        obs_dim: Observation dimensionality
        action_dim: Action dimensionality

    Raises:
        ValidationError: If dimensions are invalid
    """
    if not isinstance(state_dim, int) or state_dim <= 0:
        raise ValidationError(f"state_dim must be positive integer, got {state_dim}")

    if not isinstance(obs_dim, int) or obs_dim <= 0:
        raise ValidationError(f"obs_dim must be positive integer, got {obs_dim}")

    if not isinstance(action_dim, int) or action_dim <= 0:
        raise ValidationError(f"action_dim must be positive integer, got {action_dim}")

    # Reasonable limits to prevent memory issues
    MAX_DIM = 10000
    if state_dim > MAX_DIM:
        raise ValidationError(f"state_dim {state_dim} exceeds maximum {MAX_DIM}")

    if obs_dim > MAX_DIM:
        raise ValidationError(f"obs_dim {obs_dim} exceeds maximum {MAX_DIM}")

    if action_dim > MAX_DIM:
        raise ValidationError(f"action_dim {action_dim} exceeds maximum {MAX_DIM}")


def validate_hyperparameters(**kwargs) -> None:
    """
    Validate common hyperparameters.

    Args:
        **kwargs: Hyperparameters to validate

    Raises:
        ValidationError: If any hyperparameter is invalid
    """
    for name, value in kwargs.items():
        if name in ['learning_rate', 'temperature']:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValidationError(f"{name} must be positive number, got {value}")

        elif name in ['horizon', 'max_iterations', 'n_samples']:
            if not isinstance(value, int) or value <= 0:
                raise ValidationError(f"{name} must be positive integer, got {value}")

        elif name in ['convergence_threshold']:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValidationError(f"{name} must be positive number, got {value}")


def validate_inputs(**kwargs) -> None:
    """
    General input validation function.

    Args:
        **kwargs: Named inputs to validate

    Raises:
        ValidationError: If any input is invalid
    """
    for name, value in kwargs.items():
        if value is None:
            continue  # Allow None values

        if name.endswith('_dim'):
            if not isinstance(value, int) or value <= 0:
                raise ValidationError(f"{name} must be positive integer, got {value}")

        elif name.endswith('_array') or name in ['observation', 'action', 'state']:
            if isinstance(value, np.ndarray):
                validate_array(value, name)

        elif name.endswith('_matrix') or name in ['covariance', 'precision']:
            if isinstance(value, np.ndarray):
                validate_matrix(value, name)


def handle_errors(error_types: Tuple = (Exception,),
                 default_return=None,
                 log_errors: bool = True):
    """
    Decorator for robust error handling.

    Args:
        error_types: Tuple of exception types to handle
        default_return: Value to return on error (if not re-raising)
        log_errors: Whether to log errors

    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                if log_errors:
                    logger = get_unified_logger()
                    logger.log_error(f"Error in {func.__name__}: {str(e)}")

                if default_return is not None:
                    return default_return
                else:
                    raise
            except Exception as e:
                if log_errors:
                    logger = get_unified_logger()
                    logger.log_error(f"Unexpected error in {func.__name__}: {str(e)}", component="advanced_validation")
                raise

        return wrapper
    return decorator


def safe_divide(numerator: np.ndarray, denominator: np.ndarray,
               epsilon: float = 1e-8) -> np.ndarray:
    """
    Safely divide arrays with numerical stability.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        epsilon: Small value to add for stability

    Returns:
        Division result with numerical stability
    """
    return numerator / (denominator + epsilon)


def safe_log(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Safely compute logarithm with numerical stability.

    Args:
        x: Input array
        epsilon: Small value to clamp minimum

    Returns:
        Logarithm with numerical stability
    """
    return np.log(np.maximum(x, epsilon))


def clip_values(x: np.ndarray, min_val: Optional[float] = None,
               max_val: Optional[float] = None) -> np.ndarray:
    """
    Clip values to valid range.

    Args:
        x: Input array
        min_val: Minimum value (None to skip)
        max_val: Maximum value (None to skip)

    Returns:
        Clipped array
    """
    result = x.copy()

    if min_val is not None:
        result = np.maximum(result, min_val)

    if max_val is not None:
        result = np.minimum(result, max_val)

    return result


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")

    # Validate required fields (basic validation)
    required_fields = ['agent', 'environment']  # Basic requirements

    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Configuration missing required field: {field}")

    # Validate agent configuration
    if 'agent' in config:
        agent_config = config['agent']
        if isinstance(agent_config, dict):
            # Validate dimensions
            if 'state_dim' in agent_config:
                validate_dimensions(
                    agent_config.get('state_dim', 4),
                    agent_config.get('obs_dim', 8),
                    agent_config.get('action_dim', 2)
                )


class UnifiedValidationInterface:
    """
    Unified validation interface for the Active Inference framework.

    Provides a single, consistent interface for all validation operations across
    the framework, including input validation, security checks, performance
    validation, and health monitoring.
    """

    _instance: Optional['UnifiedValidationInterface'] = None
    _initialized = False

    def __new__(cls) -> 'UnifiedValidationInterface':
        """Singleton pattern for unified validation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the unified validation interface."""
        if not self._initialized:
            self._advanced_validator: Optional[AdvancedValidator] = None
            self._health_monitor: Optional[HealthMonitor] = None
            self._security_monitor: Optional[SecurityMonitor] = None
            self._robust_framework: Optional[RobustActiveInferenceFramework] = None
            self._global_config = {
                'strict_mode': True,
                'enable_health_monitoring': True,
                'enable_security_monitoring': True,
                'health_check_interval': 10.0,
                'max_memory_mb': 1000,
                'max_data_size_mb': 100,
                'security_level': 'medium'
            }
            self._initialized = True

    def configure(self, **config) -> 'UnifiedValidationInterface':
        """
        Configure the unified validation interface.

        Args:
            **config: Configuration options including:
                - strict_mode: Whether to use strict validation (raise exceptions)
                - enable_health_monitoring: Whether to enable health monitoring
                - enable_security_monitoring: Whether to enable security monitoring
                - health_check_interval: Health check interval in seconds
                - max_memory_mb: Maximum memory usage before alerts
                - max_data_size_mb: Maximum data size for security checks
                - security_level: Security validation strictness ('low', 'medium', 'high')

        Returns:
            Self for method chaining
        """
        self._global_config.update(config)

        # Reinitialize with new configuration
        self._initialize_validation_system()

        return self

    def _initialize_validation_system(self):
        """Initialize the validation system with current configuration."""
        # Initialize advanced validator
        self._advanced_validator = AdvancedValidator(strict_mode=self._global_config['strict_mode'])

        # Initialize health monitor if enabled
        if self._global_config['enable_health_monitoring']:
            self._health_monitor = HealthMonitor(self._global_config['health_check_interval'])
            self._health_monitor.start_monitoring()
        else:
            self._health_monitor = None

        # Initialize security monitor if enabled
        if self._global_config['enable_security_monitoring']:
            self._security_monitor = SecurityMonitor()
        else:
            self._security_monitor = None

        # Initialize robust framework
        self._robust_framework = RobustActiveInferenceFramework()

    def validate(self, data: Any, validation_type: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Unified validation method.

        Args:
            data: Data to validate
            validation_type: Type of validation ('array', 'model', 'agent', 'environment', 'performance')
            context: Additional validation context

        Returns:
            ValidationResult with validation outcome
        """
        if not self._advanced_validator:
            self._initialize_validation_system()

        return self._advanced_validator.validate_comprehensive(data, validation_type, context or {})

    def validate_array(self, arr: np.ndarray, name: str = "array",
                      expected_shape: Optional[Tuple] = None,
                      expected_dtype: Optional[np.dtype] = None,
                      min_val: Optional[float] = None, max_val: Optional[float] = None,
                      allow_nan: bool = False, allow_inf: bool = False) -> None:
        """
        Validate numpy array properties.

        Args:
            arr: Array to validate
            name: Name for error messages
            expected_shape: Expected shape
            expected_dtype: Expected data type
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_nan: Whether NaN values are allowed
            allow_inf: Whether infinite values are allowed

        Raises:
            ValidationError: If validation fails
        """
        validate_array(arr, name, expected_shape, expected_dtype, min_val, max_val, allow_nan, allow_inf)

    def validate_matrix(self, matrix: np.ndarray, name: str = "matrix",
                       square: bool = False, positive_definite: bool = False,
                       symmetric: bool = False) -> None:
        """
        Validate matrix properties.

        Args:
            matrix: Matrix to validate
            name: Name for error messages
            square: Whether matrix must be square
            positive_definite: Whether matrix must be positive definite
            symmetric: Whether matrix must be symmetric

        Raises:
            ValidationError: If validation fails
        """
        validate_matrix(matrix, name, square, positive_definite, symmetric)

    def validate_belief_state(self, beliefs_dict: Dict[str, Any]) -> None:
        """
        Validate belief state dictionary.

        Args:
            beliefs_dict: Belief state to validate

        Raises:
            ValidationError: If validation fails
        """
        validate_belief_state(beliefs_dict)

    def validate_dimensions(self, state_dim: int, obs_dim: int, action_dim: int) -> None:
        """
        Validate dimension parameters.

        Args:
            state_dim: State dimensionality
            obs_dim: Observation dimensionality
            action_dim: Action dimensionality

        Raises:
            ValidationError: If validation fails
        """
        validate_dimensions(state_dim, obs_dim, action_dim)

    def validate_hyperparameters(self, **kwargs) -> None:
        """
        Validate hyperparameters.

        Args:
            **kwargs: Hyperparameters to validate

        Raises:
            ValidationError: If validation fails
        """
        validate_hyperparameters(**kwargs)

    def validate_inputs(self, **kwargs) -> None:
        """
        General input validation.

        Args:
            **kwargs: Inputs to validate

        Raises:
            ValidationError: If validation fails
        """
        validate_inputs(**kwargs)

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration dictionary.

        Args:
            config: Configuration to validate

        Raises:
            ValidationError: If validation fails
        """
        validate_config(config)

    def validate_security(self, input_data: str, source_ip: str = None) -> ValidationResult:
        """
        Validate input for security threats.

        Args:
            input_data: Input data to check
            source_ip: Source IP address

        Returns:
            ValidationResult with security validation outcome
        """
        if not self._security_monitor:
            # Fallback to basic security validation
            from .advanced_validation import SecurityValidator
            validator = SecurityValidator()
            is_valid = validator.validate_input(input_data)
            return ValidationResult(
                is_valid=is_valid,
                errors=["Security validation failed"] if not is_valid else []
            )

        return self._security_monitor.validate_input_security(input_data, source_ip)

    def safe_operation(self, operation_name: str):
        """
        Decorator for safe operation execution with error handling.

        Args:
            operation_name: Name of the operation for logging

        Returns:
            Decorator function
        """
        def decorator(func):
            return robust_execution(max_retries=2, fallback_value=None)(func)
        return decorator

    def handle_errors(self, error_types: Tuple = (Exception,),
                     default_return=None, log_errors: bool = True):
        """
        Error handling decorator.

        Args:
            error_types: Exception types to handle
            default_return: Default return value on error
            log_errors: Whether to log errors

        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    if log_errors:
                        get_unified_logger().error(f"Error in {func.__name__}: {str(e)}")

                    if default_return is not None:
                        return default_return
                    else:
                        raise
                except Exception as e:
                    if log_errors:
                        get_unified_logger().error(f"Unexpected error in {func.__name__}: {str(e)}")
                    raise

            return wrapper
        return decorator

    def safe_divide(self, numerator: np.ndarray, denominator: np.ndarray,
                   epsilon: float = 1e-8) -> np.ndarray:
        """
        Safely divide arrays with numerical stability.

        Args:
            numerator: Numerator array
            denominator: Denominator array
            epsilon: Stability epsilon

        Returns:
            Safely divided result
        """
        return safe_divide(numerator, denominator, epsilon)

    def safe_log(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Safely compute logarithm.

        Args:
            x: Input array
            epsilon: Stability epsilon

        Returns:
            Safe logarithm result
        """
        return safe_log(x, epsilon)

    def clip_values(self, x: np.ndarray, min_val: Optional[float] = None,
                   max_val: Optional[float] = None) -> np.ndarray:
        """
        Clip array values to valid range.

        Args:
            x: Input array
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Clipped array
        """
        return clip_values(x, min_val, max_val)

    def execute_safe_agent_operation(self, agent: Any, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Safely execute agent operation with comprehensive error handling.

        Args:
            agent: Agent instance
            operation: Operation name
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Operation result dictionary
        """
        if not self._robust_framework:
            self._initialize_validation_system()

        return self._robust_framework.safe_agent_operation(agent, operation, *args, **kwargs)

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.

        Returns:
            Health status dictionary
        """
        if not self._health_monitor:
            return {'status': 'health_monitoring_disabled'}

        health_summary = self._health_monitor.get_health_summary()

        # Add validation system health
        if self._advanced_validator:
            validation_stats = self._advanced_validator.get_validation_statistics()
            health_summary['validation_stats'] = validation_stats

        # Add security system health
        if self._security_monitor:
            security_summary = self._security_monitor.get_security_summary()
            health_summary['security_stats'] = security_summary

        return health_summary

    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get validation system statistics.

        Returns:
            Validation statistics dictionary
        """
        if not self._advanced_validator:
            return {'status': 'validation_system_not_initialized'}

        return self._advanced_validator.get_validation_statistics()

    def get_security_statistics(self) -> Dict[str, Any]:
        """
        Get security monitoring statistics.

        Returns:
            Security statistics dictionary
        """
        if not self._security_monitor:
            return {'status': 'security_monitoring_disabled'}

        return self._security_monitor.get_security_summary()

    def record_request_time(self, request_time: float):
        """
        Record request timing for health monitoring.

        Args:
            request_time: Request duration in seconds
        """
        if self._health_monitor:
            self._health_monitor.record_request_time(request_time)

    def record_error(self):
        """Record an error occurrence for health monitoring."""
        if self._health_monitor:
            self._health_monitor.record_error()

    def add_health_alert_callback(self, callback: Callable[[str, HealthMetrics], None]):
        """
        Add callback for health alerts.

        Args:
            callback: Callback function for health alerts
        """
        if self._health_monitor:
            self._health_monitor.add_alert_callback(callback)

    def is_ip_blocked(self, source_ip: str) -> bool:
        """
        Check if an IP address is blocked.

        Args:
            source_ip: IP address to check

        Returns:
            True if IP is blocked
        """
        if self._security_monitor:
            return self._security_monitor.is_ip_blocked(source_ip)
        return False

    def unblock_ip(self, source_ip: str):
        """
        Unblock an IP address.

        Args:
            source_ip: IP address to unblock
        """
        if self._security_monitor:
            self._security_monitor.unblock_ip(source_ip)

    def shutdown(self):
        """Shutdown the unified validation interface."""
        if self._health_monitor:
            self._health_monitor.stop_monitoring()

        # Clear all components
        self._advanced_validator = None
        self._health_monitor = None
        self._security_monitor = None
        self._robust_framework = None
        self._initialized = False
        UnifiedValidationInterface._instance = None


# Global unified validation instance
_unified_validation_interface: Optional[UnifiedValidationInterface] = None


def get_unified_validator() -> UnifiedValidationInterface:
    """Get the unified validation interface instance."""
    global _unified_validation_interface
    if _unified_validation_interface is None:
        _unified_validation_interface = UnifiedValidationInterface()
        # Initialize with defaults
        _unified_validation_interface._initialize_validation_system()
    return _unified_validation_interface


# Backward compatibility - create a default instance
_unified_validator = get_unified_validator()


def validate_secure(data: Any, validation_type: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Convenience function for secure validation."""
    return _unified_validator.validate(data, validation_type, context)


def validate_secure_array(arr: np.ndarray, **kwargs):
    """Convenience function for secure array validation."""
    return _unified_validator.validate_array(arr, **kwargs)


def safe_execute_agent_operation(agent: Any, operation: str, *args, **kwargs) -> Dict[str, Any]:
    """Convenience function for safe agent operations."""
    return _unified_validator.execute_safe_agent_operation(agent, operation, *args, **kwargs)


# Add backward compatibility imports
__all__ = [
    # Unified validation interface
    'UnifiedValidationInterface',
    'get_unified_validator',
    'validate_secure',
    'validate_secure_array',
    'safe_execute_agent_operation',

    # Advanced validation classes
    'AdvancedValidator',
    'HealthMonitor',
    'SecurityMonitor',
    'RobustActiveInferenceFramework',
    'ValidationResult',
    'HealthMetrics',
    'SecurityEvent',
    'ValidationException',
    'SecurityException',
    'PerformanceException',
    'robust_execution',
    'error_recovery_context',

    # Backward compatibility classes and functions
    'ValidationError',
    'ActiveInferenceError',
    'ModelError',
    'InferenceError',
    'PlanningError',
    'SecurityValidator',
    'AdvancedInputValidator',
    'validate_array',
    'validate_matrix',
    'validate_probability_distribution',
    'validate_belief_state',
    'validate_dimensions',
    'validate_hyperparameters',
    'validate_inputs',
    'validate_config',
    'handle_errors',
    'safe_divide',
    'safe_log',
    'clip_values'
]
