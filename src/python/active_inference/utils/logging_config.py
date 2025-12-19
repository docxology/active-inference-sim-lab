"""
Comprehensive logging configuration for Active Inference components.

This module provides structured logging, monitoring, and telemetry capabilities
for tracking agent performance, errors, and system health.
"""

import logging
import logging.handlers
import json
import traceback
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogCategory(Enum):
    """Log category enumeration for structured logging."""
    AGENT = "agent"
    INFERENCE = "inference"
    PLANNING = "planning"
    ENVIRONMENT = "environment"
    MODEL = "model"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """Structured log entry for JSON logging."""
    timestamp: str
    level: str
    category: str
    component: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Remove None values for cleaner logs
        return {k: v for k, v in result.items() if v is not None}


class StructuredLogger:
    """
    Enhanced logger with structured logging, metrics, and monitoring capabilities.
    
    Features:
    - JSON structured logging
    - Performance metrics tracking
    - Error aggregation and reporting
    - Session-based logging
    - Configurable output formats and destinations
    """
    
    def __init__(self,
                 name: str,
                 log_level: LogLevel = LogLevel.INFO,
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 enable_json: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 session_id: Optional[str] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_level: Minimum log level
            log_file: Optional file path for logging
            enable_console: Whether to log to console
            enable_json: Whether to use JSON formatting
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            session_id: Optional session identifier
        """
        self.name = name
        self.log_level = log_level
        self.enable_json = enable_json
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.value)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Setup formatters
        if enable_json:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level.value)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level.value)
            self.logger.addHandler(file_handler)
        
        # Metrics tracking
        self._metrics = {
            'log_counts': {level.name: 0 for level in LogLevel},
            'error_counts': {},
            'performance_metrics': {},
            'start_time': time.time()
        }
        self._lock = threading.Lock()
        
        # Don't propagate to parent loggers to avoid duplicates
        self.logger.propagate = False
    
    def _log_structured(self,
                       level: LogLevel,
                       category: LogCategory,
                       message: str,
                       data: Optional[Dict[str, Any]] = None,
                       error: Optional[Exception] = None,
                       agent_id: Optional[str] = None):
        """Log structured entry with metadata."""
        try:
            # Update metrics
            with self._lock:
                self._metrics['log_counts'][level.name] += 1
                
                if error:
                    error_type = type(error).__name__
                    self._metrics['error_counts'][error_type] = \
                        self._metrics['error_counts'].get(error_type, 0) + 1
            
            # Create log entry
            entry = LogEntry(
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                level=level.name,
                category=category.value,
                component=self.name,
                message=message,
                data=data,
                error=str(error) if error else None,
                stack_trace=traceback.format_exc() if error else None,
                agent_id=agent_id,
                session_id=self.session_id
            )
            
            # Log the entry
            if self.enable_json:
                # Pass the structured data to the JSON formatter
                extra = {'structured_data': entry.to_dict()}
                self.logger.log(level.value, message, extra=extra)
            else:
                # Standard text logging
                log_msg = f"[{category.value}] {message}"
                if data:
                    log_msg += f" | Data: {data}"
                if error:
                    log_msg += f" | Error: {error}"
                
                self.logger.log(level.value, log_msg)
                
        except Exception as e:
            # Fallback logging if structured logging fails
            self.logger.error(f"Structured logging failed: {e}")
            self.logger.log(level.value, message)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
              data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """Log debug message."""
        self._log_structured(LogLevel.DEBUG, category, message, data, agent_id=agent_id)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM,
             data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """Log info message."""
        self._log_structured(LogLevel.INFO, category, message, data, agent_id=agent_id)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """Log warning message."""
        self._log_structured(LogLevel.WARNING, category, message, data, agent_id=agent_id)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM,
              data: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None,
              agent_id: Optional[str] = None):
        """Log error message."""
        self._log_structured(LogLevel.ERROR, category, message, data, error, agent_id)
    
    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                 data: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None,
                 agent_id: Optional[str] = None):
        """Log critical message."""
        self._log_structured(LogLevel.CRITICAL, category, message, data, error, agent_id)
    
    def log_performance(self, operation: str, duration: float, 
                       data: Optional[Dict[str, Any]] = None,
                       agent_id: Optional[str] = None):
        """Log performance metrics."""
        with self._lock:
            if operation not in self._metrics['performance_metrics']:
                self._metrics['performance_metrics'][operation] = []
            self._metrics['performance_metrics'][operation].append(duration)
        
        perf_data = {'operation': operation, 'duration': duration}
        if data:
            perf_data.update(data)
        
        self.info(f"Performance: {operation} took {duration:.4f}s", 
                 LogCategory.PERFORMANCE, perf_data, agent_id)
    
    def log_agent_step(self, agent_id: str, step_count: int, 
                      free_energy: float, reward: float,
                      additional_data: Optional[Dict[str, Any]] = None):
        """Log agent step information."""
        data = {
            'step_count': step_count,
            'free_energy': free_energy,
            'reward': reward
        }
        if additional_data:
            data.update(additional_data)
        
        self.debug(f"Agent step {step_count}", LogCategory.AGENT, data, agent_id)
    
    def log_inference_update(self, agent_id: str, belief_entropy: float,
                           observation_likelihood: float,
                           additional_data: Optional[Dict[str, Any]] = None):
        """Log inference update information."""
        data = {
            'belief_entropy': belief_entropy,
            'observation_likelihood': observation_likelihood
        }
        if additional_data:
            data.update(additional_data)
        
        self.debug("Belief update", LogCategory.INFERENCE, data, agent_id)
    
    def log_planning_decision(self, agent_id: str, selected_action: np.ndarray,
                            expected_free_energy: float, n_candidates: int,
                            additional_data: Optional[Dict[str, Any]] = None):
        """Log planning decision information."""
        data = {
            'selected_action': selected_action.tolist() if isinstance(selected_action, np.ndarray) else selected_action,
            'expected_free_energy': expected_free_energy,
            'n_candidates': n_candidates
        }
        if additional_data:
            data.update(additional_data)
        
        self.debug("Action planned", LogCategory.PLANNING, data, agent_id)
    
    def log_security_event(self, event_type: str, severity: str, 
                          description: str, data: Optional[Dict[str, Any]] = None):
        """Log security-related events."""
        security_data = {
            'event_type': event_type,
            'severity': severity,
            'description': description
        }
        if data:
            security_data.update(data)
        
        level = LogLevel.WARNING if severity == 'medium' else LogLevel.ERROR
        self._log_structured(level, LogCategory.SECURITY, 
                           f"Security event: {event_type}", security_data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logging and performance metrics."""
        with self._lock:
            # Calculate performance statistics
            perf_stats = {}
            for operation, durations in self._metrics['performance_metrics'].items():
                if durations:
                    perf_stats[operation] = {
                        'count': len(durations),
                        'mean': np.mean(durations),
                        'std': np.std(durations),
                        'min': np.min(durations),
                        'max': np.max(durations),
                        'total': np.sum(durations)
                    }
            
            return {
                'session_id': self.session_id,
                'uptime': time.time() - self._metrics['start_time'],
                'log_counts': self._metrics['log_counts'].copy(),
                'error_counts': self._metrics['error_counts'].copy(),
                'performance_stats': perf_stats
            }
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics = {
                'log_counts': {level.name: 0 for level in LogLevel},
                'error_counts': {},
                'performance_metrics': {},
                'start_time': time.time()
            }


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        try:
            # Check if we have structured data
            if hasattr(record, 'structured_data'):
                log_entry = record.structured_data
            else:
                # Create basic structured entry
                log_entry = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'component': record.name,
                    'message': record.getMessage(),
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_entry['error'] = str(record.exc_info[1])
                    log_entry['stack_trace'] = self.formatException(record.exc_info)
            
            return json.dumps(log_entry, default=str)
            
        except Exception as e:
            # Fallback to standard formatting if JSON fails
            return f"JSON_FORMAT_ERROR: {str(e)} | {record.getMessage()}"


class PerformanceTimer:
    """Context manager for timing operations and logging performance."""
    
    def __init__(self, logger: StructuredLogger, operation: str, 
                 agent_id: Optional[str] = None, 
                 additional_data: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.operation = operation
        self.agent_id = agent_id
        self.additional_data = additional_data or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.log_performance(
                self.operation, duration, self.additional_data, self.agent_id
            )


def setup_global_logging(log_level: LogLevel = LogLevel.INFO,
                        log_dir: Optional[str] = None,
                        enable_console: bool = True,
                        enable_json: bool = True) -> StructuredLogger:
    """
    Setup global logging configuration for the Active Inference framework.
    
    Args:
        log_level: Global log level
        log_dir: Directory for log files (None to disable file logging)
        enable_console: Whether to enable console logging
        enable_json: Whether to use JSON formatting
        
    Returns:
        Main application logger
    """
    # Setup main application logger
    log_file = None
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = str(log_path / "active_inference.log")
    
    main_logger = StructuredLogger(
        "active_inference",
        log_level=log_level,
        log_file=log_file,
        enable_console=enable_console,
        enable_json=enable_json
    )
    
    # Setup component-specific loggers
    component_loggers = {}
    
    for category in LogCategory:
        component_log_file = None
        if log_dir:
            component_log_file = str(Path(log_dir) / f"{category.value}.log")
        
        component_loggers[category.value] = StructuredLogger(
            f"active_inference.{category.value}",
            log_level=log_level,
            log_file=component_log_file,
            enable_console=False,  # Only main logger logs to console
            enable_json=enable_json
        )
    
    # Store loggers globally for easy access
    global _global_loggers
    _global_loggers = {
        'main': main_logger,
        **component_loggers
    }
    
    main_logger.info("Global logging initialized", LogCategory.SYSTEM, {
        'log_level': log_level.name,
        'log_dir': log_dir,
        'enable_console': enable_console,
        'enable_json': enable_json
    })
    
    return main_logger


def get_logger(component: str = "main") -> StructuredLogger:
    """Get logger for specific component."""
    global _global_loggers
    if '_global_loggers' not in globals() or not _global_loggers:
        # Initialize with defaults if not setup
        setup_global_logging()
    
    # If component doesn't exist, create a basic logger
    if component not in _global_loggers:
        if 'main' not in _global_loggers:
            setup_global_logging()
        _global_loggers[component] = _global_loggers['main']
    
    return _global_loggers[component]


class PerformanceMonitor:
    """
    Comprehensive performance monitor for tracking operation metrics.

    Similar to PerformanceTimer but with more detailed metrics, history tracking,
    and error monitoring.
    """

    def __init__(self, component_name: str, logger: Optional[StructuredLogger] = None):
        """
        Initialize performance monitor.

        Args:
            component_name: Name of the component being monitored
            logger: Optional structured logger for automatic logging
        """
        self.component_name = component_name
        self.logger = logger or get_logger("performance")
        self.metrics = {
            'call_count': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0,
            'last_call_time': None,
            'start_time': time.time()
        }
        self.history = []
        self._lock = threading.Lock()

    def measure(self, operation_name: str = "default"):
        """
        Context manager for measuring operation performance.

        Args:
            operation_name: Name of the operation being measured

        Returns:
            Context manager
        """
        return _PerformanceMeasurement(self, operation_name)

    def _update_metrics(self, elapsed: float, error_occurred: bool) -> None:
        """Update performance metrics."""
        with self._lock:
            self.metrics['call_count'] += 1
            self.metrics['total_time'] += elapsed
            self.metrics['average_time'] = self.metrics['total_time'] / self.metrics['call_count']
            self.metrics['min_time'] = min(self.metrics['min_time'], elapsed)
            self.metrics['max_time'] = max(self.metrics['max_time'], elapsed)
            self.metrics['last_call_time'] = elapsed

            if error_occurred:
                self.metrics['error_count'] += 1

    def _add_to_history(self, operation_name: str, elapsed: float, error_occurred: bool) -> None:
        """Add measurement to history."""
        with self._lock:
            self.history.append({
                'timestamp': time.time(),
                'operation': operation_name,
                'duration': elapsed,
                'error': error_occurred
            })

            # Keep last 1000 entries
            if len(self.history) > 1000:
                self.history.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            result = self.metrics.copy()
            result['uptime'] = time.time() - result['start_time']
            return result

    def get_history(self, n_recent: Optional[int] = None) -> List[Dict]:
        """
        Get performance history.

        Args:
            n_recent: Number of recent entries to return (None for all)

        Returns:
            List of performance records
        """
        with self._lock:
            if n_recent is None:
                return self.history.copy()
            else:
                return self.history[-n_recent:]

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self._lock:
            self.metrics = {
                'call_count': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'error_count': 0,
                'last_call_time': None,
                'start_time': time.time()
            }
            self.history.clear()

        self.logger.info(f"Performance metrics reset for {self.component_name}")


class _PerformanceMeasurement:
    """Internal context manager for performance measurement."""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.error_occurred = False

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            end_time = time.perf_counter()
            elapsed = end_time - self.start_time
            self.error_occurred = exc_type is not None

            self.monitor._update_metrics(elapsed, self.error_occurred)
            self.monitor._add_to_history(self.operation_name, elapsed, self.error_occurred)

            # Log performance if logger is available
            if self.monitor.logger:
                if elapsed > 1.0:  # Log slow operations
                    self.monitor.logger.log_performance(
                        f"{self.monitor.component_name}.{self.operation_name}",
                        elapsed,
                        {'error': self.error_occurred}
                    )


class MetricsCollector:
    """
    Collect and aggregate metrics from active inference agents.

    Provides centralized metrics collection with system monitoring and
    agent-specific metric aggregation.
    """

    def __init__(self, collection_interval: float = 10.0,
                 logger: Optional[StructuredLogger] = None):
        """
        Initialize metrics collector.

        Args:
            collection_interval: Time between metric collections (seconds)
            logger: Optional structured logger
        """
        self.collection_interval = collection_interval
        self.metrics_storage = {}
        self.registered_agents = {}
        self.logger = logger or get_logger("metrics")
        self._last_collection_time = time.time()
        self._lock = threading.Lock()

    def register_agent(self, agent_id: str, agent) -> None:
        """
        Register an agent for metric collection.

        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance to monitor
        """
        with self._lock:
            self.registered_agents[agent_id] = agent
            self.metrics_storage[agent_id] = []

        self.logger.info(f"Registered agent {agent_id} for monitoring", LogCategory.SYSTEM)

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current metrics from all registered agents.

        Returns:
            Dictionary of collected metrics
        """
        current_time = time.time()

        with self._lock:
            if current_time - self._last_collection_time < self.collection_interval:
                return {}  # Skip collection if interval hasn't passed

            collected_metrics = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'agents': {},
                'system': self._collect_system_metrics()
            }

            # Collect from each agent
            for agent_id, agent in self.registered_agents.items():
                try:
                    if hasattr(agent, 'get_statistics'):
                        agent_metrics = agent.get_statistics()
                        collected_metrics['agents'][agent_id] = agent_metrics

                        # Store in history
                        self.metrics_storage[agent_id].append({
                            'timestamp': current_time,
                            'metrics': agent_metrics
                        })

                        # Keep last 1000 entries
                        if len(self.metrics_storage[agent_id]) > 1000:
                            self.metrics_storage[agent_id].pop(0)

                except Exception as e:
                    self.logger.error(f"Error collecting metrics from agent {agent_id}: {e}",
                                    LogCategory.SYSTEM, error=e)

            self._last_collection_time = current_time
            return collected_metrics

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'cpu_count': psutil.cpu_count()
            }
        except ImportError:
            return {'note': 'psutil not available for system metrics'}

    def get_agent_history(self, agent_id: str, n_recent: Optional[int] = None) -> List[Dict]:
        """
        Get metric history for a specific agent.

        Args:
            agent_id: Agent identifier
            n_recent: Number of recent entries (None for all)

        Returns:
            List of historical metrics
        """
        with self._lock:
            if agent_id not in self.metrics_storage:
                return []

            history = self.metrics_storage[agent_id]
            if n_recent is None:
                return history.copy()
            else:
                return history[-n_recent:]

    def export_metrics(self, filepath: str, agent_id: Optional[str] = None) -> None:
        """
        Export metrics to file.

        Args:
            filepath: Path to export file
            agent_id: Specific agent to export (None for all)
        """
        if agent_id:
            data = {
                'agent_id': agent_id,
                'metrics_history': self.get_agent_history(agent_id)
            }
        else:
            data = {
                'all_agents': {
                    aid: self.get_agent_history(aid)
                    for aid in self.registered_agents.keys()
                }
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Metrics exported to {filepath}", LogCategory.SYSTEM)


class TelemetryLogger:
    """
    Log telemetry data for active inference systems.

    Provides event-based telemetry logging with structured JSON output
    and session-based organization.
    """

    def __init__(self, output_dir: str = "telemetry",
                 logger: Optional[StructuredLogger] = None):
        """
        Initialize telemetry logger.

        Args:
            output_dir: Directory to store telemetry files
            logger: Optional structured logger
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.telemetry_file = self.output_dir / f"telemetry_{self.session_id}.jsonl"

        self.logger = logger or get_logger("telemetry")
        self.logger.info(f"Telemetry logging to {self.telemetry_file}", LogCategory.SYSTEM)

        # Performance metrics
        self.event_count = 0
        self._lock = threading.Lock()

    def log_event(self, event_type: str, data: Dict[str, Any],
                  agent_id: Optional[str] = None) -> None:
        """
        Log a telemetry event.

        Args:
            event_type: Type of event (e.g., 'inference', 'planning', 'learning')
            data: Event data dictionary
            agent_id: Optional agent identifier
        """
        event = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': self.session_id,
            'event_type': event_type,
            'agent_id': agent_id,
            'data': data
        }

        with self._lock:
            # Write to file
            with open(self.telemetry_file, 'a') as f:
                f.write(json.dumps(event, default=str) + '\n')

            self.event_count += 1

    def log_inference_step(self,
                          agent_id: str,
                          observations: np.ndarray,
                          beliefs_before: Dict,
                          beliefs_after: Dict,
                          free_energy: Dict) -> None:
        """Log an inference step."""
        self.log_event('inference_step', {
            'agent_id': agent_id,
            'observation_shape': observations.shape,
            'observation_stats': {
                'mean': float(observations.mean()),
                'std': float(observations.std()),
                'min': float(observations.min()),
                'max': float(observations.max())
            },
            'beliefs_entropy_change': {
                name: float(beliefs_after[name].get('entropy', 0) -
                           beliefs_before[name].get('entropy', 0))
                for name in beliefs_before.keys()
                if name in beliefs_after
            },
            'free_energy_components': {
                'accuracy': float(free_energy.get('accuracy', 0)),
                'complexity': float(free_energy.get('complexity', 0)),
                'total': float(free_energy.get('total', 0))
            }
        }, agent_id)

    def log_planning_step(self,
                         agent_id: str,
                         selected_action: np.ndarray,
                         expected_free_energy: float,
                         n_candidates_evaluated: int) -> None:
        """Log a planning step."""
        self.log_event('planning_step', {
            'agent_id': agent_id,
            'action_shape': selected_action.shape,
            'action_stats': {
                'mean': float(selected_action.mean()),
                'std': float(selected_action.std()),
                'norm': float(np.linalg.norm(selected_action))
            },
            'expected_free_energy': float(expected_free_energy),
            'n_candidates_evaluated': int(n_candidates_evaluated)
        }, agent_id)


# Backward compatibility functions
def setup_logging(level: str = "INFO",
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> StructuredLogger:
    """
    Setup logging (backward compatibility).

    Deprecated: Use setup_global_logging instead.
    """
    log_level = getattr(LogLevel, level.upper(), LogLevel.INFO)

    # Convert old format string to enable_json flag
    enable_json = format_string is None

    logger = StructuredLogger(
        "active_inference",
        log_level=log_level,
        log_file=log_file,
        enable_json=enable_json
    )

    # Store as main logger for get_logger compatibility
    global _global_loggers
    _global_loggers['main'] = logger

    logger.info("Active Inference logging initialized (legacy setup)", LogCategory.SYSTEM)
    return logger


class UnifiedLoggingInterface:
    """
    Unified logging interface for the Active Inference framework.

    Provides a single, consistent interface for all logging operations across
    the framework, including structured logging, performance monitoring,
    metrics collection, and telemetry.
    """

    _instance: Optional['UnifiedLoggingInterface'] = None
    _initialized = False

    def __new__(cls) -> 'UnifiedLoggingInterface':
        """Singleton pattern for unified logging."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the unified logging interface."""
        if not self._initialized:
            self._loggers: Dict[str, StructuredLogger] = {}
            self._performance_monitors: Dict[str, PerformanceMonitor] = {}
            self._metrics_collector: Optional[MetricsCollector] = None
            self._telemetry_logger: Optional[TelemetryLogger] = None
            self._global_config = {
                'log_level': LogLevel.INFO,
                'enable_json': True,
                'enable_console': True,
                'enable_file': False,
                'log_file': None,
                'enable_performance_monitoring': True,
                'enable_metrics_collection': True,
                'enable_telemetry': False
            }
            self._initialized = True

    def configure(self, **config) -> 'UnifiedLoggingInterface':
        """
        Configure the unified logging interface.

        Args:
            **config: Configuration options including:
                - log_level: LogLevel enum value
                - enable_json: Whether to use JSON formatting
                - enable_console: Whether to log to console
                - enable_file: Whether to log to file
                - log_file: Path to log file
                - enable_performance_monitoring: Whether to enable performance monitoring
                - enable_metrics_collection: Whether to enable metrics collection
                - enable_telemetry: Whether to enable telemetry logging

        Returns:
            Self for method chaining
        """
        self._global_config.update(config)

        # Reinitialize with new configuration
        self._initialize_logging_system()

        return self

    def _initialize_logging_system(self):
        """Initialize the logging system with current configuration."""
        # Setup main logger
        main_logger = StructuredLogger(
            "active_inference",
            log_level=self._global_config['log_level'],
            log_file=self._global_config['log_file'] if self._global_config['enable_file'] else None,
            enable_console=self._global_config['enable_console'],
            enable_json=self._global_config['enable_json']
        )

        self._loggers['main'] = main_logger

        # Initialize metrics collector if enabled
        if self._global_config['enable_metrics_collection']:
            self._metrics_collector = MetricsCollector()
        else:
            self._metrics_collector = None

        # Initialize telemetry logger if enabled
        if self._global_config['enable_telemetry']:
            self._telemetry_logger = TelemetryLogger()
        else:
            self._telemetry_logger = None

    def get_logger(self, component: str) -> StructuredLogger:
        """
        Get a logger for a specific component.

        Args:
            component: Component name (e.g., 'agent', 'inference', 'planning')

        Returns:
            StructuredLogger instance for the component
        """
        if component not in self._loggers:
            # Create component-specific logger
            component_logger = StructuredLogger(
                f"active_inference.{component}",
                log_level=self._global_config['log_level'],
                enable_console=self._global_config['enable_console'],
                enable_json=self._global_config['enable_json']
            )
            self._loggers[component] = component_logger

        return self._loggers[component]

    def get_performance_monitor(self, component: str) -> PerformanceMonitor:
        """
        Get a performance monitor for a specific component.

        Args:
            component: Component name

        Returns:
            PerformanceMonitor instance for the component
        """
        if component not in self._performance_monitors:
            logger = self.get_logger(component) if self._global_config['enable_performance_monitoring'] else None
            self._performance_monitors[component] = PerformanceMonitor(component, logger)

        return self._performance_monitors[component]

    def log(self, level: LogLevel, message: str, component: str = "main",
            category: LogCategory = LogCategory.SYSTEM, data: Optional[Dict[str, Any]] = None,
            agent_id: Optional[str] = None):
        """
        Unified logging method.

        Args:
            level: Log level
            message: Log message
            component: Component name
            category: Log category
            data: Additional structured data
            agent_id: Optional agent identifier
        """
        logger = self.get_logger(component)
        logger._log_structured(level, category, message, data, agent_id=agent_id)

    def log_debug(self, message: str, component: str = "main",
                  category: LogCategory = LogCategory.SYSTEM,
                  data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, component, category, data, agent_id)

    def log_info(self, message: str, component: str = "main",
                 category: LogCategory = LogCategory.SYSTEM,
                 data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """Log info message."""
        self.log(LogLevel.INFO, message, component, category, data, agent_id)

    def log_warning(self, message: str, component: str = "main",
                    category: LogCategory = LogCategory.SYSTEM,
                    data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, component, category, data, agent_id)

    def log_error(self, message: str, component: str = "main",
                  category: LogCategory = LogCategory.SYSTEM,
                  data: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None,
                  agent_id: Optional[str] = None):
        """Log error message."""
        logger = self.get_logger(component)
        logger._log_structured(LogLevel.ERROR, category, message, data, error, agent_id)

    def log_performance(self, operation: str, duration: float, component: str = "performance",
                        data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """
        Log performance metrics.

        Args:
            operation: Operation name
            duration: Duration in seconds
            component: Component name
            data: Additional performance data
            agent_id: Optional agent identifier
        """
        logger = self.get_logger(component)
        logger.log_performance(operation, duration, data, agent_id)

    def log_agent_step(self, agent_id: str, step_count: int, free_energy: float, reward: float,
                       additional_data: Optional[Dict[str, Any]] = None):
        """Log agent step information."""
        logger = self.get_logger("agent")
        logger.log_agent_step(agent_id, step_count, free_energy, reward, additional_data)

    def log_inference_update(self, agent_id: str, belief_entropy: float,
                             observation_likelihood: float, additional_data: Optional[Dict[str, Any]] = None):
        """Log inference update information."""
        logger = self.get_logger("inference")
        logger.log_inference_update(agent_id, belief_entropy, observation_likelihood, additional_data)

    def log_planning_decision(self, agent_id: str, selected_action: np.ndarray,
                              expected_free_energy: float, n_candidates: int,
                              additional_data: Optional[Dict[str, Any]] = None):
        """Log planning decision information."""
        logger = self.get_logger("planning")
        logger.log_planning_decision(agent_id, selected_action, expected_free_energy,
                                   n_candidates, additional_data)

    def log_security_event(self, event_type: str, severity: str, description: str,
                          data: Optional[Dict[str, Any]] = None):
        """Log security-related events."""
        logger = self.get_logger("security")
        logger.log_security_event(event_type, severity, description, data)

    def measure_performance(self, component: str, operation: str):
        """
        Get performance measurement context manager.

        Args:
            component: Component name
            operation: Operation name

        Returns:
            Context manager for performance measurement
        """
        monitor = self.get_performance_monitor(component)
        return monitor.measure(operation)

    def register_agent_for_monitoring(self, agent_id: str, agent) -> None:
        """
        Register an agent for metrics collection.

        Args:
            agent_id: Unique agent identifier
            agent: Agent instance
        """
        if self._metrics_collector:
            self._metrics_collector.register_agent(agent_id, agent)

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current metrics from all registered agents.

        Returns:
            Dictionary of collected metrics
        """
        if self._metrics_collector:
            return self._metrics_collector.collect_metrics()
        return {}

    def log_inference_telemetry(self, agent_id: str, observations: np.ndarray,
                               beliefs_before: Dict, beliefs_after: Dict, free_energy: Dict):
        """Log inference telemetry data."""
        if self._telemetry_logger:
            self._telemetry_logger.log_inference_step(
                agent_id, observations, beliefs_before, beliefs_after, free_energy
            )

    def log_planning_telemetry(self, agent_id: str, selected_action: np.ndarray,
                              expected_free_energy: float, n_candidates: int):
        """Log planning telemetry data."""
        if self._telemetry_logger:
            self._telemetry_logger.log_planning_step(
                agent_id, selected_action, expected_free_energy, n_candidates
            )

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health information.

        Returns:
            Dictionary with health metrics and status
        """
        health_data = {
            'logging_active': True,
            'metrics_collection_active': self._metrics_collector is not None,
            'telemetry_active': self._telemetry_logger is not None,
            'performance_monitoring_active': self._global_config['enable_performance_monitoring'],
            'registered_loggers': list(self._loggers.keys()),
            'registered_monitors': list(self._performance_monitors.keys()),
            'configuration': self._global_config.copy()
        }

        # Add metrics collector stats if available
        if self._metrics_collector:
            health_data['metrics_stats'] = {
                'registered_agents': list(self._metrics_collector.registered_agents.keys()),
                'collection_interval': self._metrics_collector.collection_interval
            }

        # Add logger statistics
        logger_stats = {}
        for name, logger in self._loggers.items():
            logger_stats[name] = logger.get_metrics()

        health_data['logger_stats'] = logger_stats

        return health_data

    def shutdown(self):
        """Shutdown the unified logging interface."""
        # Clear all loggers and monitors
        self._loggers.clear()
        self._performance_monitors.clear()
        self._metrics_collector = None
        self._telemetry_logger = None
        self._initialized = False
        UnifiedLoggingInterface._instance = None


# Global unified logging instance
_unified_logging_interface: Optional[UnifiedLoggingInterface] = None


def get_unified_logger() -> UnifiedLoggingInterface:
    """Get the unified logging interface instance."""
    global _unified_logging_interface
    if _unified_logging_interface is None:
        _unified_logging_interface = UnifiedLoggingInterface()
        # Initialize with defaults
        _unified_logging_interface._initialize_logging_system()
    return _unified_logging_interface


# Backward compatibility - create a default instance
_unified_logger = get_unified_logger()


def log_debug(message: str, component: str = "main", **kwargs):
    """Convenience function for debug logging."""
    _unified_logger.log_debug(message, component, **kwargs)


def log_info(message: str, component: str = "main", **kwargs):
    """Convenience function for info logging."""
    _unified_logger.log_info(message, component, **kwargs)


def log_warning(message: str, component: str = "main", **kwargs):
    """Convenience function for warning logging."""
    _unified_logger.log_warning(message, component, **kwargs)


def log_error(message: str, component: str = "main", **kwargs):
    """Convenience function for error logging."""
    _unified_logger.log_error(message, component, **kwargs)


def log_performance(operation: str, duration: float, component: str = "performance", **kwargs):
    """Convenience function for performance logging."""
    _unified_logger.log_performance(operation, duration, component, **kwargs)


# Global instances
_default_metrics_collector: Optional[MetricsCollector] = None
_default_telemetry_logger: Optional[TelemetryLogger] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the default metrics collector instance."""
    global _default_metrics_collector
    if _default_metrics_collector is None:
        _default_metrics_collector = MetricsCollector()
    return _default_metrics_collector


def get_telemetry_logger() -> TelemetryLogger:
    """Get the default telemetry logger instance."""
    global _default_telemetry_logger
    if _default_telemetry_logger is None:
        _default_telemetry_logger = TelemetryLogger()
    return _default_telemetry_logger


# Global logger storage
_global_loggers: Dict[str, StructuredLogger] = {}


# Export key classes and functions
__all__ = [
    'StructuredLogger',
    'LogLevel',
    'LogCategory',
    'PerformanceTimer',
    'PerformanceMonitor',
    'MetricsCollector',
    'TelemetryLogger',
    'UnifiedLoggingInterface',
    'setup_global_logging',
    'setup_logging',  # Backward compatibility
    'get_logger',
    'get_metrics_collector',
    'get_telemetry_logger',
    'get_unified_logger',
    'log_debug',
    'log_info',
    'log_warning',
    'log_error',
    'log_performance'
]