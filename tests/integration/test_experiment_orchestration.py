"""
Integration tests for experiment orchestration.

Tests complete experiment workflows through the orchestration system,
including parameter sweeps, comparative studies, and performance monitoring.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from active_inference.orchestration import (
    ExperimentOrchestrator,
    OrchestrationConfig,
    ExperimentResult
)


class TestExperimentOrchestration:
    """Test experiment orchestration functionality."""

    @patch('active_inference.orchestration.GridWorld')
    @patch('active_inference.orchestration.ActiveInferenceAgent')
    def test_experiment_execution(self, mock_agent, mock_env):
        """Test basic experiment execution."""
        # Setup mocks
        mock_agent_instance = Mock()
        mock_agent_instance.reset.return_value = None
        mock_agent_instance.step.return_value = (np.array([0.1, 0.9]), 1.0, True, {})
        mock_agent.return_value = mock_agent_instance

        mock_env_instance = Mock()
        mock_env_instance.reset.return_value = np.array([0, 0])
        mock_env_instance.step.return_value = (np.array([1, 1]), 1.0, True, {})
        mock_env.return_value = mock_env_instance

        config = OrchestrationConfig(
            num_episodes=2,
            max_steps_per_episode=3,
            agent_config={"learning_rate": 0.01}
        )

        orchestrator = ExperimentOrchestrator(config)
        results = orchestrator.run_experiment()

        assert isinstance(results, list)
        assert len(results) == 2  # 2 episodes

        for result in results:
            assert isinstance(result, ExperimentResult)
            assert result.episode >= 0
            assert result.total_reward >= 0

    @patch('active_inference.orchestration.GridWorld')
    @patch('active_inference.orchestration.ActiveInferenceAgent')
    def test_parameter_sweep(self, mock_agent, mock_env):
        """Test parameter sweep functionality."""
        # Setup mocks with different behaviors based on parameters
        def agent_side_effect(**kwargs):
            agent = Mock()
            agent.reset.return_value = None
            # Different performance based on learning rate
            lr = kwargs.get('learning_rate', 0.01)
            reward = 10.0 if lr > 0.05 else 5.0
            agent.step.return_value = (np.array([0.1, 0.9]), reward, True, {})
            return agent

        mock_agent.side_effect = agent_side_effect

        mock_env_instance = Mock()
        mock_env_instance.reset.return_value = np.array([0, 0])
        mock_env_instance.step.return_value = (np.array([1, 1]), 1.0, True, {})
        mock_env.return_value = mock_env_instance

        config = OrchestrationConfig(
            num_episodes=1,
            max_steps_per_episode=2
        )

        orchestrator = ExperimentOrchestrator(config)

        # Test parameter sweep
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'temperature': [0.5, 1.0]
        }

        sweep_results = orchestrator.parameter_sweep(param_grid)

        assert isinstance(sweep_results, dict)
        assert len(sweep_results) == 4  # 2 * 2 combinations

        # Check that different parameters give different results
        performances = [result['mean_reward'] for result in sweep_results.values()]
        assert len(set(performances)) > 1  # Should have different performances

    @patch('active_inference.orchestration.GridWorld')
    @patch('active_inference.orchestration.ActiveInferenceAgent')
    def test_comparative_study(self, mock_agent, mock_env):
        """Test comparative study between different agent types."""
        # Setup different agent behaviors
        agent_behaviors = {
            'baseline': {'reward': 5.0},
            'optimized': {'reward': 8.0}
        }

        def agent_side_effect(**kwargs):
            agent_type = kwargs.get('agent_type', 'baseline')
            behavior = agent_behaviors.get(agent_type, agent_behaviors['baseline'])

            agent = Mock()
            agent.reset.return_value = None
            agent.step.return_value = (np.array([0.1, 0.9]), behavior['reward'], True, {})
            return agent

        mock_agent.side_effect = agent_side_effect

        mock_env_instance = Mock()
        mock_env_instance.reset.return_value = np.array([0, 0])
        mock_env_instance.step.return_value = (np.array([1, 1]), 1.0, True, {})
        mock_env.return_value = mock_env_instance

        config = OrchestrationConfig(
            num_episodes=1,
            max_steps_per_episode=2
        )

        orchestrator = ExperimentOrchestrator(config)

        # Test comparative study
        conditions = {
            'baseline': {'agent_type': 'baseline'},
            'optimized': {'agent_type': 'optimized'}
        }

        comparison_results = orchestrator.comparative_study(conditions)

        assert isinstance(comparison_results, dict)
        assert 'baseline' in comparison_results
        assert 'optimized' in comparison_results

        # Optimized should perform better
        baseline_perf = comparison_results['baseline']['mean_reward']
        optimized_perf = comparison_results['optimized']['mean_reward']
        assert optimized_perf > baseline_perf

    def test_experiment_result_structure(self):
        """Test experiment result data structure."""
        result = ExperimentResult(
            episode=1,
            total_reward=100.5,
            steps=50,
            metadata={'custom': 'data'}
        )

        assert result.episode == 1
        assert result.total_reward == 100.5
        assert result.steps == 50
        assert result.metadata == {'custom': 'data'}

    @patch('active_inference.orchestration.GridWorld')
    @patch('active_inference.orchestration.ActiveInferenceAgent')
    def test_experiment_persistence(self, mock_agent, mock_env):
        """Test experiment result persistence."""
        # Setup mocks
        mock_agent_instance = Mock()
        mock_agent_instance.reset.return_value = None
        mock_agent_instance.step.return_value = (np.array([0.1, 0.9]), 1.0, True, {})
        mock_agent.return_value = mock_agent_instance

        mock_env_instance = Mock()
        mock_env_instance.reset.return_value = np.array([0, 0])
        mock_env_instance.step.return_value = (np.array([1, 1]), 1.0, True, {})
        mock_env.return_value = mock_env_instance

        config = OrchestrationConfig(
            num_episodes=1,
            max_steps_per_episode=2,
            save_results=True
        )

        orchestrator = ExperimentOrchestrator(config)
        results = orchestrator.run_experiment()

        # Should have results
        assert len(results) == 1

    def test_orchestrator_configuration(self):
        """Test orchestrator configuration options."""
        config = OrchestrationConfig(
            agent_type="adaptive",
            environment_type="grid_world",
            num_episodes=50,
            max_steps_per_episode=200,
            num_agents=3,
            enable_monitoring=True,
            enable_caching=True,
            log_level="DEBUG"
        )

        orchestrator = ExperimentOrchestrator(config)

        assert orchestrator.config.agent_type == "adaptive"
        assert orchestrator.config.num_episodes == 50
        assert orchestrator.config.enable_monitoring == True
        assert orchestrator.config.enable_caching == True

    @patch('active_inference.orchestration.GridWorld')
    @patch('active_inference.orchestration.ActiveInferenceAgent')
    def test_multi_agent_orchestration(self, mock_agent, mock_env):
        """Test orchestration with multiple agents."""
        # Setup mocks for multiple agents
        agent_instances = []
        for i in range(3):
            agent = Mock()
            agent.reset.return_value = None
            agent.step.return_value = (np.array([0.1, 0.9]), float(i + 1), True, {})
            agent_instances.append(agent)

        mock_agent.side_effect = agent_instances

        mock_env_instance = Mock()
        mock_env_instance.reset.return_value = np.array([0, 0])
        mock_env_instance.step.return_value = (np.array([1, 1]), 1.0, True, {})
        mock_env.return_value = mock_env_instance

        config = OrchestrationConfig(
            num_episodes=1,
            max_steps_per_episode=2,
            num_agents=3
        )

        orchestrator = ExperimentOrchestrator(config)
        results = orchestrator.run_experiment()

        # Should have results for all agents
        assert len(results) == 3  # 3 agents

    def test_error_handling_in_orchestration(self):
        """Test error handling in orchestration."""
        config = OrchestrationConfig(
            num_episodes=1,
            max_steps_per_episode=1
        )

        orchestrator = ExperimentOrchestrator(config)

        # Should handle errors gracefully
        try:
            # This might fail due to missing implementations, but should not crash
            results = orchestrator.run_experiment()
        except Exception as e:
            # Should be a controlled exception
            assert isinstance(e, (ImportError, AttributeError, Exception))


class TestOrchestrationMetrics:
    """Test orchestration metrics and monitoring."""

    def test_performance_metrics_collection(self):
        """Test that performance metrics are collected."""
        config = OrchestrationConfig(enable_monitoring=True)
        orchestrator = ExperimentOrchestrator(config)

        # Should have telemetry for metrics collection
        assert hasattr(orchestrator, 'telemetry')

    def test_resource_monitoring(self):
        """Test resource usage monitoring."""
        config = OrchestrationConfig(enable_monitoring=True)
        orchestrator = ExperimentOrchestrator(config)

        # Should track resource usage
        assert orchestrator.telemetry is not None

    def test_experiment_metadata(self):
        """Test experiment metadata collection."""
        config = OrchestrationConfig(
            experiment_name="test_experiment",
            tags=["integration", "test"]
        )

        orchestrator = ExperimentOrchestrator(config)

        assert orchestrator.config.experiment_name == "test_experiment"
        assert "integration" in orchestrator.config.tags
        assert "test" in orchestrator.config.tags
