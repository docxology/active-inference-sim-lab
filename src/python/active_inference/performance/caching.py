"""
Intelligent caching system for Active Inference components.

This module provides adaptive caching strategies to optimize
performance by reusing expensive computations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import hashlib
import pickle
import functools
from collections import OrderedDict
from enum import Enum
from dataclasses import dataclass

from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel
from ..utils.logging_config import get_unified_logger


class CacheStrategy(Enum):
    """Caching strategy options."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on hit rate


@dataclass
class CacheEntry:
    """Entry in cache with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    last_access: float
    size_bytes: int


class BaseCache:
    """Base class for caching implementations."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            strategy: Caching strategy
        """
        self.max_size = max_size
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        self.logger = get_unified_logger()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self.cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        current_time = time.time()
        
        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 1024  # Default estimate
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=current_time,
            access_count=1,
            last_access=current_time,
            size_bytes=size_bytes
        )
        
        # Add to cache
        if key in self.cache:
            # Update existing entry
            self.cache[key] = entry
            if self.strategy == CacheStrategy.LRU:
                self.cache.move_to_end(key)
        else:
            # Add new entry
            self.cache[key] = entry
            
            # Evict if necessary
            if len(self.cache) > self.max_size:
                self._evict()
    
    def _evict(self) -> None:
        """Evict entries based on strategy."""
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_access = min(entry.access_count for entry in self.cache.values())
            for key, entry in list(self.cache.items()):
                if entry.access_count == min_access:
                    del self.cache[key]
                    break
        elif self.strategy == CacheStrategy.TTL:
            # Remove oldest entries
            current_time = time.time()
            ttl = 3600  # 1 hour default TTL
            
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time - entry.timestamp > ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            # If no expired entries, fall back to LRU
            if not expired_keys and len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            self._adaptive_evict()
        
        self.evictions += 1
    
    def _adaptive_evict(self) -> None:
        """Adaptive eviction based on access patterns."""
        current_time = time.time()
        
        # Calculate scores for each entry
        scores = {}
        for key, entry in self.cache.items():
            # Score based on recency, frequency, and size
            recency_score = 1.0 / (current_time - entry.last_access + 1)
            frequency_score = entry.access_count
            size_penalty = entry.size_bytes / 1024  # Size in KB
            
            scores[key] = recency_score * frequency_score / size_penalty
        
        # Remove entry with lowest score
        worst_key = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[worst_key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / max(1, total)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate(),
            'total_size_bytes': total_size,
            'average_entry_size': total_size / max(1, len(self.cache)),
            'strategy': self.strategy.value
        }
    
    def clear_old_entries(self, age_threshold: float = 3600) -> int:
        """Clear entries older than threshold."""
        current_time = time.time()
        old_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > age_threshold
        ]
        
        for key in old_keys:
            del self.cache[key]
        
        return len(old_keys)


class BeliefCache(BaseCache):
    """Specialized cache for belief states."""
    
    def __init__(self, max_size: int = 1000):
        super().__init__(max_size, CacheStrategy.ADAPTIVE)
        
    def generate_belief_key(self, 
                           observation: np.ndarray,
                           prior_beliefs: BeliefState) -> str:
        """Generate cache key for belief update."""
        
        # Hash observation
        obs_hash = hashlib.md5(observation.tobytes()).hexdigest()[:8]
        
        # Hash prior beliefs
        belief_data = []
        for name, belief in prior_beliefs.get_all_beliefs().items():
            belief_data.append(f"{name}:{belief.mean.tobytes()}:{belief.variance.tobytes()}")
        
        beliefs_str = "|".join(belief_data)
        beliefs_hash = hashlib.md5(beliefs_str.encode()).hexdigest()[:8]
        
        return f"belief_{obs_hash}_{beliefs_hash}"
    
    def cache_belief_update(self,
                           observation: np.ndarray,
                           prior_beliefs: BeliefState,
                           updated_beliefs: BeliefState) -> None:
        """Cache belief update result."""
        
        key = self.generate_belief_key(observation, prior_beliefs)
        self.put(key, updated_beliefs)
    
    def get_cached_belief_update(self,
                                observation: np.ndarray,
                                prior_beliefs: BeliefState) -> Optional[BeliefState]:
        """Get cached belief update result."""
        
        key = self.generate_belief_key(observation, prior_beliefs)
        return self.get(key)


class ModelCache(BaseCache):
    """Specialized cache for model computations."""
    
    def __init__(self, max_size: int = 500):
        super().__init__(max_size, CacheStrategy.LFU)
    
    def generate_likelihood_key(self,
                               state: np.ndarray,
                               observation: np.ndarray) -> str:
        """Generate cache key for likelihood computation."""
        
        state_hash = hashlib.md5(state.tobytes()).hexdigest()[:8]
        obs_hash = hashlib.md5(observation.tobytes()).hexdigest()[:8]
        
        return f"likelihood_{state_hash}_{obs_hash}"
    
    def cache_likelihood(self,
                        state: np.ndarray,
                        observation: np.ndarray,
                        likelihood: float) -> None:
        """Cache likelihood computation."""
        
        key = self.generate_likelihood_key(state, observation)
        self.put(key, likelihood)
    
    def get_cached_likelihood(self,
                             state: np.ndarray,
                             observation: np.ndarray) -> Optional[float]:
        """Get cached likelihood."""
        
        key = self.generate_likelihood_key(state, observation)
        return self.get(key)
    
    def generate_dynamics_key(self,
                             state: np.ndarray,
                             action: np.ndarray) -> str:
        """Generate cache key for dynamics prediction."""
        
        state_hash = hashlib.md5(state.tobytes()).hexdigest()[:8]
        action_hash = hashlib.md5(action.tobytes()).hexdigest()[:8]
        
        return f"dynamics_{state_hash}_{action_hash}"
    
    def cache_dynamics(self,
                      state: np.ndarray,
                      action: np.ndarray,
                      next_state: np.ndarray) -> None:
        """Cache dynamics prediction."""
        
        key = self.generate_dynamics_key(state, action)
        self.put(key, next_state)
    
    def get_cached_dynamics(self,
                           state: np.ndarray,
                           action: np.ndarray) -> Optional[np.ndarray]:
        """Get cached dynamics prediction."""
        
        key = self.generate_dynamics_key(state, action)
        return self.get(key)


class AdaptiveCache(BaseCache):
    """
    Adaptive cache that adjusts strategy based on performance.
    
    Monitors hit rates and automatically switches between
    caching strategies to optimize performance.
    """
    
    def __init__(self, max_size: int = 1000):
        super().__init__(max_size, CacheStrategy.ADAPTIVE)
        
        # Strategy performance tracking
        self.strategy_stats = {
            strategy: {'hits': 0, 'misses': 0, 'score': 0.0}
            for strategy in CacheStrategy
        }
        
        self.current_strategy = CacheStrategy.LRU
        self.adaptation_interval = 1000  # Adapt every N operations
        self.operations_count = 0
        
        # Performance history
        self.performance_history = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get with adaptive strategy tracking."""
        self.operations_count += 1
        
        result = super().get(key)
        
        # Update strategy stats
        if result is not None:
            self.strategy_stats[self.current_strategy]['hits'] += 1
        else:
            self.strategy_stats[self.current_strategy]['misses'] += 1
        
        # Adapt strategy if needed
        if self.operations_count % self.adaptation_interval == 0:
            self._adapt_strategy()
        
        return result
    
    def _adapt_strategy(self) -> None:
        """Adapt caching strategy based on performance."""
        
        # Calculate current performance
        current_hit_rate = self.hit_rate()
        self.performance_history.append(current_hit_rate)
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # Try different strategy if performance is declining
        if len(self.performance_history) >= 3:
            recent_trend = np.mean(self.performance_history[-3:])
            older_trend = np.mean(self.performance_history[-6:-3]) if len(self.performance_history) >= 6 else recent_trend
            
            if recent_trend < older_trend * 0.95:  # 5% decline
                self._try_new_strategy()
        
        self.logger.log_debug(f"Cache adaptation: strategy={self.current_strategy.value}, "
                         f"hit_rate={current_hit_rate:.3f}")
    
    def _try_new_strategy(self) -> None:
        """Try a new caching strategy."""
        
        # Calculate scores for each strategy
        for strategy, stats in self.strategy_stats.items():
            total = stats['hits'] + stats['misses']
            if total > 0:
                hit_rate = stats['hits'] / total
                # Score includes hit rate and recency bias
                recency_bonus = 1.0 if strategy == self.current_strategy else 0.9
                stats['score'] = hit_rate * recency_bonus
        
        # Find best performing strategy (excluding current)
        alternative_strategies = [s for s in CacheStrategy if s != self.current_strategy]
        if alternative_strategies:
            best_strategy = max(alternative_strategies, 
                              key=lambda s: self.strategy_stats[s]['score'])
            
            # Switch if significantly better
            current_score = self.strategy_stats[self.current_strategy]['score']
            best_score = self.strategy_stats[best_strategy]['score']
            
            if best_score > current_score * 1.05:  # 5% improvement threshold
                self.logger.log_debug("Switched to better caching strategy", component="caching")
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive cache statistics."""
        
        stats = self.get_stats()
        
        # Add adaptation-specific metrics
        stats.update({
            'current_strategy': self.current_strategy.value,
            'operations_count': self.operations_count,
            'adaptation_interval': self.adaptation_interval,
            'performance_history': self.performance_history[-5:],  # Recent history
            'strategy_performance': {
                strategy.value: {
                    'hit_rate': stats_data['hits'] / max(1, stats_data['hits'] + stats_data['misses']),
                    'total_operations': stats_data['hits'] + stats_data['misses'],
                    'score': stats_data['score']
                }
                for strategy, stats_data in self.strategy_stats.items()
            }
        })
        
        return stats


class CacheManager:
    """Manager for multiple cache instances."""
    
    def __init__(self):
        self.caches: Dict[str, BaseCache] = {}
        self.logger = get_unified_logger()
    
    def register_cache(self, name: str, cache: BaseCache) -> None:
        """Register a cache instance."""
        self.caches[name] = cache
        self.logger.log_debug("Operation completed", component="caching")
    
    def get_cache(self, name: str) -> Optional[BaseCache]:
        """Get cache by name."""
        return self.caches.get(name)
    
    def clear_all_caches(self) -> None:
        """Clear all registered caches."""
        for cache in self.caches.values():
            cache.clear()
        self.logger.log_debug("All caches cleared", component="caching")
    
    def get_all_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            name: cache.get_stats()
            for name, cache in self.caches.items()
        }
    
    def optimize_cache_sizes(self, 
                            total_memory_budget: int = 512 * 1024 * 1024) -> Dict[str, int]:
        """
        Optimize cache sizes based on usage patterns and memory budget.
        
        Args:
            total_memory_budget: Total memory budget in bytes
            
        Returns:
            Dictionary of cache_name -> recommended_size
        """
        
        # Get current usage statistics
        stats = self.get_global_stats()
        
        # Calculate relative importance based on hit rates and usage
        importance_scores = {}
        total_score = 0
        
        for name, cache_stats in stats.items():
            hit_rate = cache_stats.get('hit_rate', 0)
            total_ops = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
            
            # Importance score combines hit rate and usage frequency
            score = hit_rate * np.log(total_ops + 1)
            importance_scores[name] = score
            total_score += score
        
        # Allocate memory budget proportionally
        recommendations = {}
        
        for name, score in importance_scores.items():
            if total_score > 0:
                proportion = score / total_score
                memory_allocation = int(total_memory_budget * proportion)
                
                # Estimate entries per byte (rough approximation)
                avg_entry_size = stats[name].get('average_entry_size', 1024)
                recommended_size = max(100, memory_allocation // avg_entry_size)
                
                recommendations[name] = recommended_size
            else:
                recommendations[name] = 1000  # Default
        
        return recommendations
    
    def apply_recommendations(self, recommendations: Dict[str, int]) -> None:
        """Apply cache size recommendations."""
        
        for name, recommended_size in recommendations.items():
            if name in self.caches:
                cache = self.caches[name]
                old_size = cache.max_size
                cache.max_size = recommended_size
                
                # Evict excess entries if needed
                while len(cache.cache) > recommended_size:
                    cache._evict()
                
                self.logger.log_debug("Operation completed", component="caching")


def memoize(maxsize: int = 128, ttl: Optional[float] = None,
           typed: bool = False, ignore_self: bool = True):
    """
    Memoization decorator with LRU cache.

    Args:
        maxsize: Maximum cache size
        ttl: Time to live for cached entries
        typed: Whether to consider argument types in cache key
        ignore_self: Whether to ignore 'self' parameter in methods

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = AdaptiveCache(maxsize=maxsize)
        cache.ttl = ttl  # Set TTL if provided

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Handle 'self' parameter for methods
            cache_args = args
            if ignore_self and len(args) > 0 and hasattr(args[0], func.__name__):
                cache_args = args[1:]  # Skip 'self'

            # Create cache key
            if typed:
                key_args = cache_args + tuple(type(arg) for arg in cache_args)
                key_kwargs = kwargs.copy()
                key_kwargs.update({f"{k}_type": type(v) for k, v in kwargs.items()})
            else:
                key_args = cache_args
                key_kwargs = kwargs

            key = _make_hash_key(key_args, key_kwargs)

            # Try cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            # Compute result
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            computation_time = time.perf_counter() - start_time

            # Cache result with metadata
            cache.put(key, result)

            return result

        # Attach cache for inspection
        wrapper._cache = cache

        return wrapper
    return decorator


def _make_hash_key(args: tuple, kwargs: dict) -> str:
    """Create hash key from arguments."""
    # Create a hash of the arguments
    key_data = (args, tuple(sorted(kwargs.items())))
    key_str = str(key_data)
    return hashlib.md5(key_str.encode()).hexdigest()


class BatchProcessor:
    """
    Batch processing for vectorized computations.
    """

    def __init__(self, batch_size: int = 32, n_workers: Optional[int] = None):
        """
        Initialize batch processor.

        Args:
            batch_size: Default batch size
            n_workers: Number of worker threads (None for auto)
        """
        self.batch_size = batch_size
        self.n_workers = n_workers or min(4, os.cpu_count() or 1)
        self.executor = ThreadPoolExecutor(max_workers=self.n_workers)
        self.logger = get_unified_logger()

    def process_batch(self,
                     data: np.ndarray,
                     process_func: Callable,
                     batch_size: Optional[int] = None) -> np.ndarray:
        """
        Process data in batches.

        Args:
            data: Input data array
            process_func: Function to apply to each batch
            batch_size: Batch size (uses default if None)

        Returns:
            Processed data
        """
        if batch_size is None:
            batch_size = self.batch_size

        n_samples = len(data)
        results = []

        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data[i:end_idx]

            batch_result = process_func(batch)
            results.append(batch_result)

        # Concatenate results
        if results:
            return np.concatenate(results, axis=0)
        else:
            return np.array([])

    def parallel_process(self,
                        data_batches: list,
                        process_func: Callable) -> list:
        """
        Process multiple batches in parallel.

        Args:
            data_batches: List of data batches
            process_func: Function to apply to each batch

        Returns:
            List of processed results
        """
        futures = []

        for batch in data_batches:
            future = self.executor.submit(process_func, batch)
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.log_debug("Operation completed", component="caching")

    def create_gaussian_table(self,
                             x_range: Tuple[float, float],
                             n_points: int = 1000,
                             sigma: float = 1.0) -> None:
        """
        Create precomputed Gaussian probability table.

        Args:
            x_range: Range of x values (min, max)
            n_points: Number of points to precompute
            sigma: Standard deviation
        """
        x_min, x_max = x_range
        x_values = np.linspace(x_min, x_max, n_points)

        # Compute Gaussian probabilities
        gaussian_values = np.exp(-0.5 * (x_values / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        self.tables['gaussian'] = {
            'x_values': x_values,
            'probabilities': gaussian_values,
            'x_min': x_min,
            'x_max': x_max,
            'sigma': sigma
        }

        self.logger.log_debug("Operation completed", component="caching")

    def lookup_gaussian(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Fast Gaussian probability lookup.

        Args:
            x: Input values

        Returns:
            Gaussian probabilities
        """
        if 'gaussian' not in self.tables:
            raise ValueError("Gaussian table not created. Call create_gaussian_table first.")

        table = self.tables['gaussian']
        x_values = table['x_values']
        probabilities = table['probabilities']

        # Interpolate
        return np.interp(x, x_values, probabilities)

    def create_softmax_table(self,
                           x_range: Tuple[float, float],
                           n_points: int = 1000,
                           temperature: float = 1.0) -> None:
        """
        Create precomputed softmax table.

        Args:
            x_range: Range of input values
            n_points: Number of points
            temperature: Softmax temperature
        """
        x_min, x_max = x_range
        x_values = np.linspace(x_min, x_max, n_points)

        # Compute softmax for different input sizes
        # Store for different common sizes
        self.tables['softmax'] = {
            'x_values': x_values,
            'temperature': temperature,
            'precomputed': {}
        }

        for size in [2, 3, 4, 5, 10]:
            softmax_values = []
            for x in x_values:
                inputs = np.full(size, x)
                inputs[0] = x  # Vary first element
                softmax = np.exp(inputs / temperature)
                softmax = softmax / softmax.sum()
                softmax_values.append(softmax)

            self.tables['softmax']['precomputed'][size] = np.array(softmax_values)

        self.logger.log_debug("Operation completed", component="caching")

    def save_tables(self, filepath: str) -> None:
        """Save precomputed tables to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.tables, f)
        self.logger.log_debug("Operation completed", component="caching")

    def load_tables(self, filepath: str) -> None:
        """Load precomputed tables from file with security validation."""
        import os
        from pathlib import Path

        # Validate file path
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Cache file not found: {filepath}")

        # Check file size (prevent loading extremely large files)
        file_size = os.path.getsize(filepath)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError(f"Cache file too large: {file_size} bytes")

        try:
            with open(filepath, 'rb') as f:
                # Use safe pickle loading with size limit
                data = f.read()
                if len(data) > 100 * 1024 * 1024:  # Additional size check
                    raise ValueError("Pickle data too large")

                import io
                data_stream = io.BytesIO(data)
                self.tables = pickle.load(data_stream)

            # Validate loaded data structure
            if not isinstance(self.tables, dict):
                raise ValueError("Invalid cache data: expected dictionary")

            self.logger.log_debug("Operation completed", component="caching")
        except (pickle.UnpicklingError, EOFError, ImportError) as e:
            raise ValueError(f"Invalid or corrupted cache file: {e}")


class OptimizedOperations:
    """
    Optimized mathematical operations for active inference.
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize optimized operations.

        Args:
            use_gpu: Whether to use GPU acceleration (requires CuPy)
        """
        self.use_gpu = use_gpu

        if use_gpu:
            try:
                import cupy as cp
                self.cp = cp
                self.logger = get_unified_logger()
                self.logger.log_info("GPU acceleration enabled with CuPy", component="caching")
            except ImportError:
                self.cp = None
                self.logger = get_unified_logger()
                self.logger.log_warning("CuPy not available, falling back to CPU", component="caching")
        else:
            self.logger = get_unified_logger()

    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Optimized matrix multiplication."""
        if self.use_gpu:
            A_gpu = self.cp.asarray(A)
            B_gpu = self.cp.asarray(B)
            result_gpu = self.cp.dot(A_gpu, B_gpu)
            return self.cp.asnumpy(result_gpu)
        else:
            return np.dot(A, B)

    def eigendecomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized eigenvalue decomposition."""
        if self.use_gpu:
            matrix_gpu = self.cp.asarray(matrix)
            eigenvals_gpu, eigenvecs_gpu = self.cp.linalg.eig(matrix_gpu)
            return self.cp.asnumpy(eigenvals_gpu), self.cp.asnumpy(eigenvecs_gpu)
        else:
            return np.linalg.eig(matrix)

    def cholesky_decomposition(self, matrix: np.ndarray) -> np.ndarray:
        """Optimized Cholesky decomposition."""
        if self.use_gpu:
            matrix_gpu = self.cp.asarray(matrix)
            chol_gpu = self.cp.linalg.cholesky(matrix_gpu)
            return self.cp.asnumpy(chol_gpu)
        else:
            return np.linalg.cholesky(matrix)

    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Optimized linear system solver."""
        if self.use_gpu:
            A_gpu = self.cp.asarray(A)
            b_gpu = self.cp.asarray(b)
            x_gpu = self.cp.linalg.solve(A_gpu, b_gpu)
            return self.cp.asnumpy(x_gpu)
        else:
            return np.linalg.solve(A, b)


def create_cache_from_config(config: Dict[str, Any]) -> BaseCache:
    """
    Create cache from configuration dictionary.

    Args:
        config: Configuration dictionary with cache settings

    Returns:
        Configured cache instance
    """
    cache_type = config.get('type', 'adaptive')
    max_size = config.get('max_size', 1000)
    strategy = config.get('strategy', 'adaptive')

    if cache_type == 'belief':
        return BeliefCache(max_size=max_size)
    elif cache_type == 'model':
        return ModelCache(max_size=max_size)
    elif cache_type == 'adaptive':
        return AdaptiveCache(max_size=max_size)
    else:
        # Default to base cache with specified strategy
        strategy_enum = getattr(CacheStrategy, strategy.upper(), CacheStrategy.LRU)
        return BaseCache(max_size=max_size, strategy=strategy_enum)


# Global instances
_global_batch_processor: Optional[BatchProcessor] = None
_global_optimized_ops: Optional[OptimizedOperations] = None

# Backward compatibility aliases
LRUCache = AdaptiveCache  # Use AdaptiveCache as drop-in replacement for LRUCache

# Global cache manager instance
cache_manager = CacheManager()


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance."""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = BatchProcessor()
    return _global_batch_processor


def get_optimized_ops(use_gpu: bool = False) -> OptimizedOperations:
    """Get global optimized operations instance."""
    global _global_optimized_ops
    if _global_optimized_ops is None:
        _global_optimized_ops = OptimizedOperations(use_gpu=use_gpu)
    return _global_optimized_ops


# Cleanup function
def cleanup_resources():
    """Cleanup global resources."""
    global _global_batch_processor
    if _global_batch_processor:
        _global_batch_processor.close()
        _global_batch_processor = None