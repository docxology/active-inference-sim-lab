"""
Integration tests for orchestration workflows.

Tests complete end-to-end orchestration of Active Inference components
including agents, environments, monitoring, and performance optimization.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

from active_inference.orchestration import (
    OrchestrationConfig,
    ExperimentOrchestrator,
    AgentOrchestrator,
    create_orchestrator,
    run_quick_experiment
)


class TestOrchestrationWorkflows:
    """Test orchestration workflow integration."""

    def test_orchestration_config_creation(self):
        """Test creation of orchestration configuration."""
        config = OrchestrationConfig(
            agent_type="active_inference",
            environment_type="grid_world",
            num_episodes=10,
            max_steps_per_episode=50
        )

        assert config.agent_type == "active_inference"
        assert config.environment_type == "grid_world"
        assert config.num_episodes == 10
        assert config.max_steps_per_episode == 50

    def test_agent_orchestrator_initialization(self):
        """Test agent orchestrator initialization."""
        config = OrchestrationConfig()
        orchestrator = AgentOrchestrator(config)

        assert orchestrator.config == config
        assert hasattr(orchestrator, 'logger')
        assert hasattr(orchestrator, 'validator')

    @patch('active_inference.orchestration.GridWorld')
    @patch('active_inference.orchestration.ActiveInferenceAgent')
    def test_experiment_orchestrator_creation(self, mock_agent, mock_env):
        """Test experiment orchestrator with mocked components."""
        # Setup mocks
        mock_agent_instance = Mock()
        mock_agent_instance.reset.return_value = None
        mock_agent_instance.step.return_value = (np.array([0.5]), 1.0, False, {})
        mock_agent.return_value = mock_agent_instance

        mock_env_instance = Mock()
        mock_env_instance.reset.return_value = np.array([0, 0])
        mock_env_instance.step.return_value = (np.array([1, 1]), 1.0, False, {})
        mock_env.return_value = mock_env_instance

        config = OrchestrationConfig(
            num_episodes=2,
            max_steps_per_episode=3
        )

        orchestrator = ExperimentOrchestrator(config)

        # Test that orchestrator was created successfully
        assert orchestrator.config == config
        assert hasattr(orchestrator, 'logger')
        assert hasattr(orchestrator, 'validator')

    def test_create_orchestrator_function(self):
        """Test the create_orchestrator factory function."""
        config = OrchestrationConfig()
        orchestrator = create_orchestrator(config)

        assert isinstance(orchestrator, AgentOrchestrator)

    def test_unified_logging_integration(self):
        """Test that orchestration uses unified logging."""
        config = OrchestrationConfig()
        orchestrator = AgentOrchestrator(config)

        # Check that logger has unified logging methods
        assert hasattr(orchestrator.logger, 'log_info')
        assert hasattr(orchestrator.logger, 'log_error')
        assert hasattr(orchestrator.logger, 'log_warning')

    def test_unified_validation_integration(self):
        """Test that orchestration uses unified validation."""
        config = OrchestrationConfig()
        orchestrator = AgentOrchestrator(config)

        # Check that validator has unified validation methods
        assert hasattr(orchestrator.validator, 'validate')
        assert hasattr(orchestrator.validator, 'validate_array')
        assert hasattr(orchestrator.validator, 'validate_secure')

    @patch('active_inference.orchestration.GridWorld')
    def test_environment_integration(self, mock_env):
        """Test environment integration through orchestration."""
        mock_env_instance = Mock()
        mock_env_instance.reset.return_value = np.array([0, 0])
        mock_env.return_value = mock_env_instance

        config = OrchestrationConfig(environment_type="grid_world")
        orchestrator = AgentOrchestrator(config)

        # Test that environment can be accessed
        assert orchestrator.environment is not None

    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        config = OrchestrationConfig(enable_monitoring=True)
        orchestrator = AgentOrchestrator(config)

        # Check that telemetry collector is available
        assert hasattr(orchestrator, 'telemetry')

    def test_caching_integration(self):
        """Test caching integration."""
        config = OrchestrationConfig(enable_caching=True)
        orchestrator = AgentOrchestrator(config)

        # Check that cache manager is available
        assert hasattr(orchestrator, 'cache_manager')

    def test_configuration_validation(self):
        """Test configuration validation."""
        config = OrchestrationConfig()

        # Test valid configuration
        assert config.num_episodes > 0
        assert config.max_steps_per_episode > 0

        # Test invalid configuration would be caught
        with pytest.raises(ValueError):
            config.num_episodes = -1

    def test_orchestrator_lifecycle(self):
        """Test orchestrator lifecycle management."""
        config = OrchestrationConfig()
        orchestrator = AgentOrchestrator(config)

        # Test initialization
        assert orchestrator._is_initialized

        # Test shutdown (should not raise errors)
        orchestrator.shutdown()

    @patch('active_inference.orchestration.run_quick_experiment')
    def test_quick_experiment_function(self, mock_run):
        """Test quick experiment function."""
        mock_run.return_value = {"success": True, "results": []}

        result = run_quick_experiment()

        assert result["success"] == True
        mock_run.assert_called_once()


class TestOrchestrationErrorHandling:
    """Test error handling in orchestration."""

    def test_orchestrator_graceful_shutdown(self):
        """Test graceful shutdown on errors."""
        config = OrchestrationConfig()
        orchestrator = AgentOrchestrator(config)

        # Should handle shutdown gracefully
        orchestrator.shutdown()
        orchestrator.shutdown()  # Second shutdown should be safe

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        # Should create orchestrator with defaults for missing config
        orchestrator = create_orchestrator()

        assert orchestrator is not None
        assert isinstance(orchestrator, AgentOrchestrator)


class TestOrchestrationPerformance:
    """Test orchestration performance characteristics."""

    def test_orchestrator_initialization_performance(self):
        """Test orchestrator initialization is fast."""
        start_time = time.time()

        config = OrchestrationConfig()
        orchestrator = AgentOrchestrator(config)

        init_time = time.time() - start_time

        # Should initialize in reasonable time
        assert init_time < 1.0  # Less than 1 second

    def test_memory_usage_tracking(self):
        """Test memory usage is tracked."""
        config = OrchestrationConfig()
        orchestrator = AgentOrchestrator(config)

        # Should have memory tracking capabilities
        assert hasattr(orchestrator, 'telemetry')
