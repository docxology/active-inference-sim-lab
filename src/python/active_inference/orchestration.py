"""
Orchestration Module for Active Inference Systems.

This module provides high-level orchestration capabilities for coordinating
Active Inference experiments, deployments, and system management. It serves
as the central coordination point for all framework components.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import time
from pathlib import Path
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import atexit

from .utils.logging_config import get_unified_logger
from .utils.advanced_validation import get_unified_validator
from .performance.caching import cache_manager
from .core import ActiveInferenceAgent, AdaptiveActiveInferenceAgent
try:
    from .environments import GymWrapper, ActiveInferenceGridWorld as GridWorld
except ImportError:
    # Fallback if gym not available
    from .environments import ActiveInferenceGridWorld as GridWorld
    GymWrapper = None
from .monitoring import AgentTelemetry as TelemetryCollector


@dataclass
class OrchestrationConfig:
    """Configuration for orchestration operations."""

    # Agent configuration
    agent_type: str = "active_inference"
    agent_config: Dict[str, Any] = field(default_factory=dict)

    # Environment configuration
    environment_type: str = "grid_world"
    environment_config: Dict[str, Any] = field(default_factory=dict)

    # Experiment configuration
    num_episodes: int = 100
    max_steps_per_episode: int = 1000
    num_agents: int = 1
    parallel_execution: bool = False
    max_workers: int = 4

    # Monitoring and logging
    enable_monitoring: bool = True
    enable_telemetry: bool = True
    log_level: str = "INFO"
    results_dir: str = "results"

    # Performance configuration
    enable_caching: bool = True
    enable_performance_monitoring: bool = True

    # Validation configuration
    enable_validation: bool = True
    strict_validation: bool = False


@dataclass
class ExperimentResult:
    """Result of an orchestrated experiment."""

    experiment_id: str
    agent_id: str
    total_episodes: int
    total_steps: int
    total_reward: float
    average_reward: float
    success_rate: float
    execution_time: float
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    telemetry_data: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    """
    Orchestrator for managing Active Inference agents.

    Provides lifecycle management, configuration, and coordination
    of multiple agents in experiments and deployments.
    """

    def __init__(self, config: OrchestrationConfig):
        """
        Initialize the agent orchestrator.

        Args:
            config: Orchestration configuration
        """
        self.config = config
        self.logger = get_unified_logger()
        self.validator = get_unified_validator()

        # Initialize components
        self.agents: Dict[str, Any] = {}
        self.environments: Dict[str, Any] = {}
        self.telemetry_collectors: Dict[str, TelemetryCollector] = {}
        self.executor: Optional[ThreadPoolExecutor] = None

        # Results storage
        self.results: List[ExperimentResult] = []
        self.experiment_counter = 0

        # Setup
        self._initialize_components()
        self._setup_signal_handlers()

    def _initialize_components(self):
        """Initialize orchestration components."""
        # Configure logging
        self.logger.configure(
            log_level=self.config.log_level,
            enable_performance_monitoring=self.config.enable_performance_monitoring
        )

        # Configure validation
        self.validator.configure(
            strict_mode=self.config.strict_validation,
            enable_health_monitoring=self.config.enable_monitoring,
            enable_security_monitoring=True
        )

        # Setup thread pool if parallel execution enabled
        if self.config.parallel_execution:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Create results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.log_warning("Received shutdown signal, cleaning up...")
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self.shutdown)

    def create_agent(self, agent_id: str, **agent_kwargs) -> Any:
        """
        Create and register an agent.

        Args:
            agent_id: Unique identifier for the agent
            **agent_kwargs: Additional agent configuration

        Returns:
            Created agent instance
        """
        # Merge configurations
        config = {**self.config.agent_config, **agent_kwargs}

        # Validate configuration
        if self.config.enable_validation:
            validation_result = self.validator.validate(config, "model")
            if not validation_result.is_valid:
                raise ValueError(f"Invalid agent configuration: {validation_result.errors}")

        # Create agent based on type
        if self.config.agent_type == "active_inference":
            agent = ActiveInferenceAgent(**config)
        elif self.config.agent_type == "adaptive":
            agent = AdaptiveActiveInferenceAgent(**config)
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")

        # Register agent
        self.agents[agent_id] = agent

        # Setup telemetry if enabled
        if self.config.enable_telemetry:
            telemetry = TelemetryCollector()
            telemetry.start()
            self.telemetry_collectors[agent_id] = telemetry

            # Register agent for monitoring
            self.logger.register_agent_for_monitoring(agent_id, agent)

        self.logger.log_info(f"Created agent {agent_id}", "orchestration")
        return agent

    def create_environment(self, env_id: str, **env_kwargs) -> Any:
        """
        Create and register an environment.

        Args:
            env_id: Unique identifier for the environment
            **env_kwargs: Additional environment configuration

        Returns:
            Created environment instance
        """
        # Merge configurations
        config = {**self.config.environment_config, **env_kwargs}

        # Create environment based on type
        if self.config.environment_type == "grid_world":
            environment = GridWorld(**config)
        elif self.config.environment_type.startswith("gym"):
            # Extract gym environment name
            gym_env_name = self.config.environment_type.split(":", 1)[1] if ":" in self.config.environment_type else "CartPole-v1"
            environment = GymWrapper(gym_env_name, **config)
        else:
            raise ValueError(f"Unknown environment type: {self.config.environment_type}")

        # Register environment
        self.environments[env_id] = environment

        self.logger.log_info(f"Created environment {env_id}", "orchestration")
        return environment

    def run_experiment(self, agent_id: str, env_id: str,
                      experiment_name: Optional[str] = None) -> ExperimentResult:
        """
        Run an experiment with the specified agent and environment.

        Args:
            agent_id: ID of the agent to use
            env_id: ID of the environment to use
            experiment_name: Optional name for the experiment

        Returns:
            ExperimentResult with comprehensive results
        """
        experiment_id = experiment_name or f"experiment_{self.experiment_counter}"
        self.experiment_counter += 1

        start_time = time.time()

        try:
            agent = self.agents[agent_id]
            environment = self.environments[env_id]

            self.logger.log_info(f"Starting experiment {experiment_id} with agent {agent_id} in environment {env_id}",
                               "orchestration")

            # Run episodes
            total_reward = 0.0
            total_steps = 0
            episode_rewards = []
            episode_lengths = []

            for episode in range(self.config.num_episodes):
                episode_reward = 0.0
                episode_steps = 0
                done = False

                # Reset environment
                observation = environment.reset()

                while not done and episode_steps < self.config.max_steps_per_episode:
                    # Agent action
                    with self.logger.measure_performance("orchestration", "agent_action"):
                        action = agent.act(observation)

                    # Environment step
                    with self.logger.measure_performance("orchestration", "environment_step"):
                        next_obs, reward, done, info = environment.step(action)

                    # Update agent
                    with self.logger.measure_performance("orchestration", "agent_update"):
                        agent.update(observation, action, reward, next_obs)

                    episode_reward += reward
                    episode_steps += 1
                    observation = next_obs

                    # Record telemetry
                    if self.config.enable_telemetry and agent_id in self.telemetry_collectors:
                        self.logger.log_inference_telemetry(
                            agent_id, np.array([observation]), {}, {}, {}
                        )

                # Record episode results
                total_reward += episode_reward
                total_steps += episode_steps
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)

                self.logger.log_info(f"Episode {episode + 1}/{self.config.num_episodes}: "
                                   f"Reward = {episode_reward:.2f}, Steps = {episode_steps}",
                                   "orchestration")

            # Calculate statistics
            average_reward = total_reward / self.config.num_episodes
            success_rate = sum(1 for r in episode_rewards if r > 0) / self.config.num_episodes

            execution_time = time.time() - start_time

            # Get performance metrics
            performance_metrics = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'average_episode_length': np.mean(episode_lengths),
                'reward_std': np.std(episode_rewards),
                'steps_per_second': total_steps / execution_time
            }

            # Get telemetry data
            telemetry_data = []
            if agent_id in self.telemetry_collectors:
                # Note: In a real implementation, you'd collect actual telemetry data
                telemetry_data = [{"experiment_id": experiment_id, "agent_id": agent_id}]

            # Create result
            result = ExperimentResult(
                experiment_id=experiment_id,
                agent_id=agent_id,
                total_episodes=self.config.num_episodes,
                total_steps=total_steps,
                total_reward=total_reward,
                average_reward=average_reward,
                success_rate=success_rate,
                execution_time=execution_time,
                performance_metrics=performance_metrics,
                telemetry_data=telemetry_data,
                metadata={
                    'environment_id': env_id,
                    'config': self.config.__dict__
                }
            )

            self.results.append(result)

            # Save results
            self._save_experiment_result(result)

            self.logger.log_info(f"Completed experiment {experiment_id}: "
                               f"Avg Reward = {average_reward:.2f}, Success Rate = {success_rate:.2%}",
                               "orchestration")

            return result

        except Exception as e:
            error_msg = f"Experiment {experiment_id} failed: {str(e)}"
            self.logger.log_error(error_msg, "orchestration", error=e)

            # Create error result
            result = ExperimentResult(
                experiment_id=experiment_id,
                agent_id=agent_id,
                total_episodes=0,
                total_steps=0,
                total_reward=0.0,
                average_reward=0.0,
                success_rate=0.0,
                execution_time=time.time() - start_time,
                errors=[error_msg]
            )

            return result

    def run_parallel_experiments(self, experiments: List[Dict[str, str]]) -> List[ExperimentResult]:
        """
        Run multiple experiments in parallel.

        Args:
            experiments: List of experiment configurations, each with 'agent_id' and 'env_id'

        Returns:
            List of experiment results
        """
        if not self.config.parallel_execution or not self.executor:
            # Fall back to sequential execution
            return [self.run_experiment(exp['agent_id'], exp['env_id']) for exp in experiments]

        self.logger.log_info(f"Running {len(experiments)} experiments in parallel", "orchestration")

        # Submit experiments to thread pool
        futures = []
        for exp in experiments:
            future = self.executor.submit(self.run_experiment, exp['agent_id'], exp['env_id'])
            futures.append(future)

        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.log_error(f"Parallel experiment failed: {e}", "orchestration", error=e)
                # Create error result
                error_result = ExperimentResult(
                    experiment_id="parallel_error",
                    agent_id="unknown",
                    total_episodes=0,
                    total_steps=0,
                    total_reward=0.0,
                    average_reward=0.0,
                    success_rate=0.0,
                    execution_time=0.0,
                    errors=[str(e)]
                )
                results.append(error_result)

        return results

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.

        Returns:
            Health status dictionary
        """
        health_data = {
            'orchestrator_status': 'active',
            'registered_agents': list(self.agents.keys()),
            'registered_environments': list(self.environments.keys()),
            'active_telemetry_collectors': list(self.telemetry_collectors.keys()),
            'parallel_execution': self.config.parallel_execution,
            'total_experiments_run': len(self.results)
        }

        # Add component health
        health_data.update(self.logger.get_system_health())
        health_data.update(self.validator.get_health_status())

        # Add cache statistics
        if self.config.enable_caching:
            health_data['cache_stats'] = cache_manager.get_global_stats()

        return health_data

    def get_experiment_results(self, experiment_id: Optional[str] = None) -> Union[ExperimentResult, List[ExperimentResult]]:
        """
        Get experiment results.

        Args:
            experiment_id: Specific experiment ID, or None for all results

        Returns:
            Single result or list of all results
        """
        if experiment_id:
            for result in self.results:
                if result.experiment_id == experiment_id:
                    return result
            return None
        else:
            return self.results.copy()

    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to file."""
        try:
            results_file = Path(self.config.results_dir) / f"{result.experiment_id}.json"

            # Convert to serializable format
            result_dict = {
                'experiment_id': result.experiment_id,
                'agent_id': result.agent_id,
                'total_episodes': result.total_episodes,
                'total_steps': result.total_steps,
                'total_reward': result.total_reward,
                'average_reward': result.average_reward,
                'success_rate': result.success_rate,
                'execution_time': result.execution_time,
                'performance_metrics': result.performance_metrics,
                'telemetry_data': result.telemetry_data,
                'errors': result.errors,
                'metadata': result.metadata
            }

            with open(results_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)

        except Exception as e:
            self.logger.log_error(f"Failed to save experiment result: {e}", "orchestration", error=e)

    def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        self.logger.log_info("Shutting down orchestrator...", "orchestration")

        # Stop telemetry collectors
        for telemetry in self.telemetry_collectors.values():
            try:
                telemetry.stop()
            except Exception as e:
                self.logger.log_error(f"Error stopping telemetry: {e}", "orchestration")

        # Shutdown thread pool
        if self.executor:
            self.executor.shutdown(wait=True)

        # Clear resources
        self.agents.clear()
        self.environments.clear()
        self.telemetry_collectors.clear()

        # Shutdown unified interfaces
        try:
            self.logger.shutdown()
            self.validator.shutdown()
        except Exception as e:
            self.logger.log_error(f"Error during orchestrator shutdown: {e}", component="orchestration")

        self.logger.log_info("Orchestrator shutdown complete", "orchestration")


class ExperimentOrchestrator:
    """
    High-level experiment orchestration for complex Active Inference studies.

    Provides advanced orchestration capabilities for running multiple experiments,
    parameter sweeps, and comparative studies.
    """

    def __init__(self, base_config: OrchestrationConfig):
        """
        Initialize experiment orchestrator.

        Args:
            base_config: Base configuration for experiments
        """
        self.base_config = base_config
        self.logger = get_unified_logger()
        self.agent_orchestrators: List[AgentOrchestrator] = []

    def run_parameter_sweep(self, parameter_space: Dict[str, List[Any]],
                           num_trials: int = 3) -> List[ExperimentResult]:
        """
        Run parameter sweep experiments.

        Args:
            parameter_space: Dictionary mapping parameter names to lists of values
            num_trials: Number of trials per parameter combination

        Returns:
            List of all experiment results
        """
        self.logger.log_info(f"Starting parameter sweep with {len(parameter_space)} parameters",
                           "experiment_orchestration")

        all_results = []
        param_combinations = self._generate_parameter_combinations(parameter_space)

        for combo in param_combinations:
            for trial in range(num_trials):
                # Create configuration for this combination
                config = self._create_config_from_parameters(combo)

                # Create orchestrator with this config
                orchestrator = AgentOrchestrator(config)
                self.agent_orchestrators.append(orchestrator)

                try:
                    # Create agent and environment
                    agent = orchestrator.create_agent(f"agent_{len(all_results)}")
                    env = orchestrator.create_environment(f"env_{len(all_results)}")

                    # Run experiment
                    result = orchestrator.run_experiment(
                        f"agent_{len(all_results)}",
                        f"env_{len(all_results)}",
                        f"param_sweep_{len(all_results)}_trial_{trial}"
                    )

                    # Add parameter information to result
                    result.metadata['parameters'] = combo
                    result.metadata['trial'] = trial

                    all_results.append(result)

                except Exception as e:
                    self.logger.log_error(f"Parameter sweep experiment failed: {e}",
                                        "experiment_orchestration", error=e)

        self.logger.log_info(f"Completed parameter sweep: {len(all_results)} experiments",
                           "experiment_orchestration")

        return all_results

    def run_comparative_study(self, agent_configs: List[Dict[str, Any]],
                             env_configs: List[Dict[str, Any]]) -> Dict[str, List[ExperimentResult]]:
        """
        Run comparative study across different agent and environment configurations.

        Args:
            agent_configs: List of agent configurations to compare
            env_configs: List of environment configurations to compare

        Returns:
            Dictionary mapping configuration names to result lists
        """
        self.logger.log_info("Starting comparative study", "experiment_orchestration")

        results = {}

        for i, agent_config in enumerate(agent_configs):
            for j, env_config in enumerate(env_configs):
                config_name = f"agent_{i}_env_{j}"

                # Create configuration
                config = OrchestrationConfig(
                    agent_config=agent_config,
                    environment_config=env_config,
                    **self.base_config.__dict__
                )

                # Create orchestrator
                orchestrator = AgentOrchestrator(config)
                self.agent_orchestrators.append(orchestrator)

                try:
                    # Run experiments
                    agent = orchestrator.create_agent(f"agent_{config_name}")
                    env = orchestrator.create_environment(f"env_{config_name}")

                    result = orchestrator.run_experiment(
                        f"agent_{config_name}",
                        f"env_{config_name}",
                        config_name
                    )

                    results[config_name] = [result]

                except Exception as e:
                    self.logger.log_error(f"Comparative study experiment {config_name} failed: {e}",
                                        "experiment_orchestration", error=e)

        self.logger.log_info(f"Completed comparative study: {len(results)} configurations tested",
                           "experiment_orchestration")

        return results

    def _generate_parameter_combinations(self, parameter_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        import itertools

        keys = list(parameter_space.keys())
        values = list(parameter_space.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _create_config_from_parameters(self, parameters: Dict[str, Any]) -> OrchestrationConfig:
        """Create orchestration config from parameter combination."""
        config_dict = self.base_config.__dict__.copy()

        # Apply parameters to appropriate config sections
        for param_name, param_value in parameters.items():
            if param_name.startswith('agent_'):
                if 'agent_config' not in config_dict:
                    config_dict['agent_config'] = {}
                config_dict['agent_config'][param_name[6:]] = param_value  # Remove 'agent_' prefix
            elif param_name.startswith('env_'):
                if 'environment_config' not in config_dict:
                    config_dict['environment_config'] = {}
                config_dict['environment_config'][param_name[4:]] = param_value  # Remove 'env_' prefix
            else:
                config_dict[param_name] = param_value

        return OrchestrationConfig(**config_dict)

    def get_study_summary(self, results: Union[List[ExperimentResult], Dict[str, List[ExperimentResult]]]) -> Dict[str, Any]:
        """
        Generate summary statistics for a study.

        Args:
            results: Experiment results to summarize

        Returns:
            Summary statistics dictionary
        """
        if isinstance(results, dict):
            # Flatten results
            all_results = []
            for result_list in results.values():
                all_results.extend(result_list)
        else:
            all_results = results

        if not all_results:
            return {'status': 'no_results'}

        # Calculate summary statistics
        total_experiments = len(all_results)
        successful_experiments = sum(1 for r in all_results if not r.errors)
        avg_reward = np.mean([r.average_reward for r in all_results if not r.errors])
        avg_success_rate = np.mean([r.success_rate for r in all_results if not r.errors])
        total_execution_time = sum(r.execution_time for r in all_results)

        return {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / total_experiments,
            'average_reward': avg_reward,
            'average_success_rate': avg_success_rate,
            'total_execution_time': total_execution_time,
            'average_execution_time': total_execution_time / total_experiments
        }

    def shutdown(self):
        """Shutdown all orchestrators."""
        for orchestrator in self.agent_orchestrators:
            try:
                orchestrator.shutdown()
            except Exception as e:
                self.logger.log_error(f"Error shutting down orchestrator: {e}", "experiment_orchestration")

        self.agent_orchestrators.clear()


# Convenience functions for easy access
def create_orchestrator(config: Optional[OrchestrationConfig] = None) -> AgentOrchestrator:
    """
    Create an agent orchestrator with default or custom configuration.

    Args:
        config: Optional custom configuration

    Returns:
        Configured AgentOrchestrator instance
    """
    if config is None:
        config = OrchestrationConfig()

    return AgentOrchestrator(config)


def run_quick_experiment(agent_type: str = "active_inference",
                        environment_type: str = "grid_world",
                        num_episodes: int = 10) -> ExperimentResult:
    """
    Run a quick experiment with default settings.

    Args:
        agent_type: Type of agent to use
        environment_type: Type of environment to use
        num_episodes: Number of episodes to run

    Returns:
        Experiment result
    """
    config = OrchestrationConfig(
        agent_type=agent_type,
        environment_type=environment_type,
        num_episodes=num_episodes,
        enable_monitoring=True,
        enable_telemetry=False  # Disable for quick experiments
    )

    orchestrator = AgentOrchestrator(config)
    agent = orchestrator.create_agent("quick_agent")
    env = orchestrator.create_environment("quick_env")

    result = orchestrator.run_experiment("quick_agent", "quick_env", "quick_experiment")

    orchestrator.shutdown()

    return result


__all__ = [
    'OrchestrationConfig',
    'ExperimentResult',
    'AgentOrchestrator',
    'ExperimentOrchestrator',
    'create_orchestrator',
    'run_quick_experiment'
]
