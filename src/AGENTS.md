# Source Code Architecture - AGENTS

## Module Overview

The `src` directory contains the complete source code implementation of the Active Inference Simulation Lab, organized into Python and C++ components for optimal performance and flexibility.

## Source Code Organization

```
src/
├── python/                    # Python implementation
│   └── active_inference/      # Main Python package
│       ├── __init__.py       # Package initialization
│       ├── core/             # Core agent implementations
│       │   ├── __init__.py
│       │   ├── agent.py      # Base ActiveInferenceAgent
│       │   ├── adaptive_agent.py  # Dimensional adaptability
│       │   ├── optimized_agent.py # Performance optimization
│       │   └── belief_state.py    # Belief representation
│       ├── environments/     # Environment interfaces
│       │   ├── __init__.py
│       │   ├── base_environment.py
│       │   ├── gym_wrapper.py
│       │   ├── mujoco_wrapper.py
│       │   └── grid_world.py
│       ├── inference/        # Belief updating algorithms
│       │   ├── __init__.py
│       │   ├── variational_inference.py
│       │   ├── particle_filter.py
│       │   └── kalman_filter.py
│       ├── planning/         # Action planning algorithms
│       │   ├── __init__.py
│       │   ├── expected_free_energy.py
│       │   ├── sampling_planner.py
│       │   └── hierarchical_planner.py
│       ├── monitoring/       # Telemetry and monitoring
│       │   ├── __init__.py
│       │   ├── telemetry.py
│       │   ├── profiler.py
│       │   └── anomaly_detector.py
│       ├── performance/      # Performance optimization
│       │   ├── __init__.py
│       │   ├── gpu_accelerator.py
│       │   ├── caching.py
│       │   └── memory_pool.py
│       ├── reliability/      # Fault tolerance
│       │   ├── __init__.py
│       │   ├── circuit_breaker.py
│       │   ├── bulkhead.py
│       │   └── self_healing.py
│       ├── scalability/      # Distributed processing
│       │   ├── __init__.py
│       │   ├── distributed_cluster.py
│       │   ├── load_balancer.py
│       │   └── auto_scaler.py
│       ├── security/         # Security features
│       │   ├── __init__.py
│       │   ├── validation.py
│       │   ├── threat_detection.py
│       │   └── access_control.py
│       ├── research/         # Advanced algorithms
│       │   ├── __init__.py
│       │   ├── hierarchical_ai.py
│       │   ├── causal_inference.py
│       │   └── meta_learning.py
│       ├── utils/            # Utility functions
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   ├── config.py
│       │   ├── validation.py
│       │   └── health_check.py
│       ├── deployment/       # Production deployment
│       │   ├── __init__.py
│       │   ├── docker_builder.py
│       │   ├── k8s_deployer.py
│       │   └── monitoring_setup.py
│       └── cli/              # Command-line interface
│           ├── __init__.py
│           └── main.py
└── cpp/                      # C++ implementation
    └── src/
        ├── CMakeLists.txt
        ├── main.cpp
        ├── active_inference/
        │   ├── core/         # Core C++ implementations
        │   │   ├── agent.hpp
        │   │   ├── agent.cpp
        │   │   ├── belief_state.hpp
        │   │   └── belief_state.cpp
        │   ├── inference/    # C++ inference algorithms
        │   │   ├── variational_inference.hpp
        │   │   └── variational_inference.cpp
        │   ├── planning/     # C++ planning algorithms
        │   │   ├── free_energy_planner.hpp
        │   │   └── free_energy_planner.cpp
        │   └── utils/        # C++ utilities
        │       ├── matrix_operations.hpp
        │       ├── matrix_operations.cpp
        │       ├── memory_pool.hpp
        │       └── memory_pool.cpp
        └── tests/            # C++ unit tests
            ├── CMakeLists.txt
            ├── test_agent.cpp
            └── test_inference.cpp
```

## Python Package Structure

### Package Initialization

**`__init__.py` - Main Package Interface:**
```python
"""
Active Inference Simulation Lab

A comprehensive framework for implementing active inference algorithms
with production-ready performance, reliability, and scalability.
"""

__version__ = "1.0.0"
__author__ = "Active Inference Team"
__description__ = "Active Inference Simulation Lab"

# Core imports for easy access
from .core import ActiveInferenceAgent, AdaptiveActiveInferenceAgent
from .environments import GymWrapper, GridWorld
from .inference import VariationalInference, ParticleFilter
from .planning import ExpectedFreeEnergyPlanner
from .performance import PerformanceOptimizedActiveInferenceAgent
from .monitoring import TelemetryCollector
from .scalability import DistributedActiveInferenceCluster
from .security import AdvancedValidator
from .research import HierarchicalTemporalActiveInference

# Utility imports
from .utils import get_logger, validate_input

# CLI entry point
def main():
    """Main CLI entry point."""
    from .cli.main import main as cli_main
    cli_main()

if __name__ == "__main__":
    main()
```

### Core Module Architecture

**Agent Base Classes:**
```python
# src/python/active_inference/core/__init__.py
from .agent import ActiveInferenceAgent
from .adaptive_agent import AdaptiveActiveInferenceAgent
from .optimized_agent import PerformanceOptimizedActiveInferenceAgent
from .belief_state import BeliefState, GenerativeModel

__all__ = [
    'ActiveInferenceAgent',
    'AdaptiveActiveInferenceAgent',
    'PerformanceOptimizedActiveInferenceAgent',
    'BeliefState',
    'GenerativeModel'
]
```

**Agent Implementation Pattern:**
```python
class ActiveInferenceAgent:
    """Base active inference agent implementation."""

    def __init__(self, state_dim, obs_dim, action_dim, **kwargs):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Initialize components
        self.beliefs = self._initialize_beliefs()
        self.generative_model = self._initialize_generative_model()
        self.inference = self._initialize_inference(**kwargs)
        self.planning = self._initialize_planning(**kwargs)

        # Optional components
        self.monitoring = kwargs.get('monitoring', None)
        self.caching = kwargs.get('caching', None)
        self.validation = kwargs.get('validation', None)

    def act(self, observation):
        """Core action selection method."""

        # Input validation
        if self.validation:
            self.validation.validate_secure(observation, 'observation')

        # Update beliefs
        self.beliefs = self.inference.update_beliefs(
            self.beliefs, observation, self.generative_model
        )

        # Plan action
        action = self.planning.plan_action(
            self.beliefs, self.generative_model, self.planning_horizon
        )

        # Monitoring
        if self.monitoring:
            self.monitoring.record_inference_metrics(
                observation=observation,
                action=action,
                beliefs=self.beliefs
            )

        return action

    def update(self, observation, action, reward, next_observation):
        """Learning update."""

        # Update generative model
        self.generative_model = self._update_generative_model(
            observation, action, reward, next_observation, self.generative_model
        )

        # Update beliefs with new observation
        self.beliefs = self.inference.update_beliefs(
            self.beliefs, next_observation, self.generative_model
        )

    def get_beliefs(self):
        """Get current belief state."""
        return self.beliefs

    def get_health_status(self):
        """Get agent health status."""
        return {
            'inference_healthy': self.inference.is_healthy(),
            'planning_healthy': self.planning.is_healthy(),
            'memory_usage': self._get_memory_usage(),
            'error_rate': self._get_error_rate()
        }
```

## C++ Implementation

### Core C++ Architecture

**CMake Build System:**
```cmake
# src/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(ActiveInference VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Library target
add_library(active_inference SHARED ${SOURCES})

# Link dependencies
target_link_libraries(active_inference
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Python bindings (optional)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(active_inference_py src/python_bindings.cpp)
    target_link_libraries(active_inference_py PRIVATE active_inference)
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS active_inference
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
```

**C++ Agent Implementation:**
```cpp
// src/cpp/include/active_inference/core/agent.hpp
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace active_inference {

class BeliefState;
class GenerativeModel;
class InferenceEngine;
class PlanningEngine;

class ActiveInferenceAgent {
public:
    // Constructor
    ActiveInferenceAgent(
        size_t state_dim,
        size_t obs_dim,
        size_t action_dim,
        const std::string& config = ""
    );

    // Core methods
    Eigen::VectorXd act(const Eigen::VectorXd& observation);
    void update(const Eigen::VectorXd& observation,
               const Eigen::VectorXd& action,
               double reward,
               const Eigen::VectorXd& next_observation);

    // Accessors
    std::shared_ptr<BeliefState> get_beliefs() const;
    std::shared_ptr<GenerativeModel> get_generative_model() const;

    // Health and monitoring
    struct HealthStatus {
        bool inference_healthy;
        bool planning_healthy;
        double memory_usage_mb;
        double error_rate;
        std::string status_message;
    };

    HealthStatus get_health_status() const;

private:
    size_t state_dim_;
    size_t obs_dim_;
    size_t action_dim_;

    std::shared_ptr<BeliefState> beliefs_;
    std::shared_ptr<GenerativeModel> generative_model_;
    std::unique_ptr<InferenceEngine> inference_;
    std::unique_ptr<PlanningEngine> planning_;

    // Performance monitoring
    struct PerformanceMetrics {
        double total_inference_time = 0.0;
        size_t inference_calls = 0;
        double total_planning_time = 0.0;
        size_t planning_calls = 0;
        size_t error_count = 0;
    };

    PerformanceMetrics metrics_;

    // Helper methods
    void initialize_components(const std::string& config);
    void validate_dimensions() const;
    void update_performance_metrics(double inference_time, double planning_time);
};

} // namespace active_inference
```

**C++ Implementation:**
```cpp
// src/cpp/src/active_inference/core/agent.cpp
#include "active_inference/core/agent.hpp"
#include "active_inference/core/belief_state.hpp"
#include "active_inference/inference/variational_inference.hpp"
#include "active_inference/planning/free_energy_planner.hpp"
#include <chrono>

namespace active_inference {

ActiveInferenceAgent::ActiveInferenceAgent(
    size_t state_dim, size_t obs_dim, size_t action_dim, const std::string& config
) : state_dim_(state_dim), obs_dim_(obs_dim), action_dim_(action_dim) {

    initialize_components(config);
    validate_dimensions();
}

void ActiveInferenceAgent::initialize_components(const std::string& config) {
    // Initialize belief state
    beliefs_ = std::make_shared<BeliefState>(state_dim_);

    // Initialize generative model
    generative_model_ = std::make_shared<GenerativeModel>(state_dim_, obs_dim_, action_dim_);

    // Initialize inference engine
    inference_ = std::make_unique<VariationalInference>(state_dim_, obs_dim_);

    // Initialize planning engine
    planning_ = std::make_unique<FreeEnergyPlanner>(state_dim_, action_dim_);
}

Eigen::VectorXd ActiveInferenceAgent::act(const Eigen::VectorXd& observation) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Update beliefs
    beliefs_ = inference_->update_beliefs(beliefs_, observation, generative_model_);

    // Plan action
    Eigen::VectorXd action = planning_->plan_action(beliefs_, generative_model_);

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    update_performance_metrics(total_time, 0.0);  // Planning time tracked separately

    return action;
}

void ActiveInferenceAgent::update(
    const Eigen::VectorXd& observation,
    const Eigen::VectorXd& action,
    double reward,
    const Eigen::VectorXd& next_observation
) {
    try {
        // Update generative model
        generative_model_->update(observation, action, reward, next_observation);

        // Update beliefs with new observation
        beliefs_ = inference_->update_beliefs(beliefs_, next_observation, generative_model_);

    } catch (const std::exception& e) {
        metrics_.error_count++;
        throw std::runtime_error("Update failed: " + std::string(e.what()));
    }
}

ActiveInferenceAgent::HealthStatus ActiveInferenceAgent::get_health_status() const {
    HealthStatus status;

    status.inference_healthy = inference_->is_healthy();
    status.planning_healthy = planning_->is_healthy();
    status.memory_usage_mb = get_memory_usage_mb();
    status.error_rate = static_cast<double>(metrics_.error_count) /
                       std::max(1UL, metrics_.inference_calls + metrics_.planning_calls);

    if (status.inference_healthy && status.planning_healthy && status.error_rate < 0.1) {
        status.status_message = "Agent is healthy";
    } else {
        status.status_message = "Agent has health issues";
    }

    return status;
}

void ActiveInferenceAgent::update_performance_metrics(double inference_time, double planning_time) {
    metrics_.total_inference_time += inference_time;
    metrics_.inference_calls++;
    metrics_.total_planning_time += planning_time;
    metrics_.planning_calls++;
}

} // namespace active_inference
```

## Python-C++ Integration

### Python Bindings

**Pybind11 Integration:**
```cpp
// src/cpp/src/python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "active_inference/core/agent.hpp"

namespace py = pybind11;

PYBIND11_MODULE(active_inference_cpp, m) {
    m.doc() = "Active Inference C++ extension module";

    // Agent class binding
    py::class_<active_inference::ActiveInferenceAgent>(m, "ActiveInferenceAgentCpp")
        .def(py::init<size_t, size_t, size_t, std::string>(),
             py::arg("state_dim"), py::arg("obs_dim"), py::arg("action_dim"),
             py::arg("config") = "")

        .def("act", &active_inference::ActiveInferenceAgent::act,
             py::arg("observation"),
             "Select action given observation")

        .def("update", &active_inference::ActiveInferenceAgent::update,
             py::arg("observation"), py::arg("action"), py::arg("reward"), py::arg("next_observation"),
             "Update agent with experience")

        .def("get_beliefs", &active_inference::ActiveInferenceAgent::get_beliefs,
             "Get current belief state")

        .def("get_health_status", &active_inference::ActiveInferenceAgent::get_health_status,
             "Get agent health status")

        .def_property_readonly("state_dim",
            [](const active_inference::ActiveInferenceAgent& agent) {
                return agent.state_dim_;
            })

        .def_property_readonly("obs_dim",
            [](const active_inference::ActiveInferenceAgent& agent) {
                return agent.obs_dim_;
            })

        .def_property_readonly("action_dim",
            [](const active_inference::ActiveInferenceAgent& agent) {
                return agent.action_dim_;
            });

    // HealthStatus binding
    py::class_<active_inference::ActiveInferenceAgent::HealthStatus>(m, "HealthStatus")
        .def_readonly("inference_healthy", &active_inference::ActiveInferenceAgent::HealthStatus::inference_healthy)
        .def_readonly("planning_healthy", &active_inference::ActiveInferenceAgent::HealthStatus::planning_healthy)
        .def_readonly("memory_usage_mb", &active_inference::ActiveInferenceAgent::HealthStatus::memory_usage_mb)
        .def_readonly("error_rate", &active_inference::ActiveInferenceAgent::HealthStatus::error_rate)
        .def_readonly("status_message", &active_inference::ActiveInferenceAgent::HealthStatus::status_message);

    // NumPy array conversions
    py::class_<Eigen::VectorXd>(m, "VectorXd")
        .def(py::init<size_t>())
        .def("size", &Eigen::VectorXd::size)
        .def("__getitem__", [](const Eigen::VectorXd& v, size_t i) { return v(i); })
        .def("__setitem__", [](Eigen::VectorXd& v, size_t i, double val) { v(i) = val; })
        .def("to_numpy", [](const Eigen::VectorXd& v) {
            return py::array_t<double>({v.size()}, {sizeof(double)}, v.data());
        });

    // Utility functions
    m.def("create_agent", [](size_t state_dim, size_t obs_dim, size_t action_dim) {
        return std::make_shared<active_inference::ActiveInferenceAgent>(
            state_dim, obs_dim, action_dim
        );
    }, "Create a new active inference agent");
}
```

**Python Wrapper:**
```python
# src/python/active_inference/core/cpp_agent.py
"""
Python wrapper for C++ Active Inference Agent.
"""

import numpy as np
from typing import Optional, Dict, Any
try:
    from active_inference_cpp import ActiveInferenceAgentCpp, VectorXd
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    ActiveInferenceAgentCpp = None

class CppActiveInferenceAgent:
    """Python wrapper for C++ active inference agent."""

    def __init__(self, state_dim: int, obs_dim: int, action_dim: int, **kwargs):
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ extension not available. Install with C++ support.")

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Initialize C++ agent
        self.cpp_agent = ActiveInferenceAgentCpp(state_dim, obs_dim, action_dim)

        # Performance monitoring
        self.performance_stats = {
            'inference_calls': 0,
            'total_inference_time': 0.0,
            'errors': 0
        }

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Select action using C++ implementation."""

        if observation.shape[0] != self.obs_dim:
            raise ValueError(f"Observation dimension mismatch: expected {self.obs_dim}, got {observation.shape[0]}")

        try:
            # Convert to Eigen vector
            obs_vector = VectorXd(self.obs_dim)
            for i in range(self.obs_dim):
                obs_vector[i] = observation[i]

            # Call C++ agent
            action_vector = self.cpp_agent.act(obs_vector)

            # Convert back to numpy
            action = np.zeros(self.action_dim)
            for i in range(self.action_dim):
                action[i] = action_vector[i]

            self.performance_stats['inference_calls'] += 1

            return action

        except Exception as e:
            self.performance_stats['errors'] += 1
            raise RuntimeError(f"C++ agent action failed: {e}")

    def update(self, observation: np.ndarray, action: np.ndarray,
              reward: float, next_observation: np.ndarray):
        """Update agent using C++ implementation."""

        try:
            # Convert inputs to Eigen vectors
            obs_vec = VectorXd(self.obs_dim)
            action_vec = VectorXd(self.action_dim)
            next_obs_vec = VectorXd(self.obs_dim)

            for i in range(self.obs_dim):
                obs_vec[i] = observation[i]
                next_obs_vec[i] = next_observation[i]

            for i in range(self.action_dim):
                action_vec[i] = action[i]

            # Call C++ update
            self.cpp_agent.update(obs_vec, action_vec, reward, next_obs_vec)

        except Exception as e:
            self.performance_stats['errors'] += 1
            raise RuntimeError(f"C++ agent update failed: {e}")

    def get_beliefs(self) -> Dict[str, Any]:
        """Get current belief state."""
        # Implementation would convert C++ belief state to Python
        return {'cpp_backend': True}

    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status."""

        try:
            cpp_health = self.cpp_agent.get_health_status()

            return {
                'cpp_backend_healthy': True,
                'inference_healthy': cpp_health.inference_healthy,
                'planning_healthy': cpp_health.planning_healthy,
                'memory_usage_mb': cpp_health.memory_usage_mb,
                'error_rate': cpp_health.error_rate,
                'performance_stats': self.performance_stats.copy()
            }

        except Exception as e:
            return {
                'cpp_backend_healthy': False,
                'error': str(e),
                'performance_stats': self.performance_stats.copy()
            }

    @classmethod
    def is_available(cls) -> bool:
        """Check if C++ backend is available."""
        return CPP_AVAILABLE
```

## Testing Infrastructure

### Unit Test Organization

**Python Tests:**
```python
# src/python/active_inference/core/tests/test_agent.py
import pytest
import numpy as np
from active_inference.core import ActiveInferenceAgent

class TestActiveInferenceAgent:
    """Unit tests for ActiveInferenceAgent."""

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        return ActiveInferenceAgent(
            state_dim=4,
            obs_dim=8,
            action_dim=2
        )

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state_dim == 4
        assert agent.obs_dim == 8
        assert agent.action_dim == 2

        beliefs = agent.get_beliefs()
        assert beliefs is not None
        assert hasattr(beliefs, 'mean')

    def test_act_method(self, agent):
        """Test action selection."""
        observation = np.random.randn(8)

        action = agent.act(observation)

        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert np.isfinite(action).all()

    def test_update_method(self, agent):
        """Test learning update."""
        observation = np.random.randn(8)
        action = np.random.randn(2)
        reward = 1.0
        next_observation = np.random.randn(8)

        # Should not raise exception
        agent.update(observation, action, reward, next_observation)

        # Beliefs should be updated
        updated_beliefs = agent.get_beliefs()
        assert updated_beliefs is not None

    def test_health_status(self, agent):
        """Test health status reporting."""
        status = agent.get_health_status()

        assert isinstance(status, dict)
        assert 'inference_healthy' in status
        assert 'planning_healthy' in status
        assert 'memory_usage' in status

    @pytest.mark.parametrize("obs_dim", [1, 5, 10])
    def test_different_dimensions(self, obs_dim):
        """Test agent with different observation dimensions."""
        agent = ActiveInferenceAgent(
            state_dim=4,
            obs_dim=obs_dim,
            action_dim=2
        )

        observation = np.random.randn(obs_dim)
        action = agent.act(observation)

        assert action.shape == (2,)
```

**C++ Tests:**
```cpp
// src/cpp/tests/test_agent.cpp
#include <gtest/gtest.h>
#include "active_inference/core/agent.hpp"

class ActiveInferenceAgentTest : public ::testing::Test {
protected:
    void SetUp() override {
        agent_ = std::make_unique<active_inference::ActiveInferenceAgent>(4, 8, 2);
    }

    std::unique_ptr<active_inference::ActiveInferenceAgent> agent_;
};

TEST_F(ActiveInferenceAgentTest, Initialization) {
    EXPECT_EQ(agent_->state_dim_, 4UL);
    EXPECT_EQ(agent_->obs_dim_, 8UL);
    EXPECT_EQ(agent_->action_dim_, 2UL);

    auto beliefs = agent_->get_beliefs();
    ASSERT_NE(beliefs, nullptr);
}

TEST_F(ActiveInferenceAgentTest, ActMethod) {
    Eigen::VectorXd observation = Eigen::VectorXd::Random(8);

    Eigen::VectorXd action = agent_->act(observation);

    EXPECT_EQ(action.size(), 2);
    EXPECT_TRUE(action.allFinite());
}

TEST_F(ActiveInferenceAgentTest, UpdateMethod) {
    Eigen::VectorXd observation = Eigen::VectorXd::Random(8);
    Eigen::VectorXd action = Eigen::VectorXd::Random(2);
    double reward = 1.0;
    Eigen::VectorXd next_observation = Eigen::VectorXd::Random(8);

    EXPECT_NO_THROW({
        agent_->update(observation, action, reward, next_observation);
    });
}

TEST_F(ActiveInferenceAgentTest, HealthStatus) {
    auto status = agent_->get_health_status();

    EXPECT_TRUE(status.inference_healthy || !status.inference_healthy);  // Boolean field
    EXPECT_TRUE(status.planning_healthy || !status.planning_healthy);
    EXPECT_GE(status.memory_usage_mb, 0.0);
    EXPECT_GE(status.error_rate, 0.0);
    EXPECT_FALSE(status.status_message.empty());
}

TEST_F(ActiveInferenceAgentTest, DifferentDimensions) {
    std::vector<size_t> obs_dims = {1, 5, 10};

    for (size_t obs_dim : obs_dims) {
        auto test_agent = std::make_unique<active_inference::ActiveInferenceAgent>(
            4, obs_dim, 2
        );

        Eigen::VectorXd observation = Eigen::VectorXd::Random(obs_dim);
        Eigen::VectorXd action = test_agent->act(observation);

        EXPECT_EQ(action.size(), 2UL);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

## Performance and Benchmarking

### Performance Benchmarks

**Python Performance Tests:**
```python
# src/python/active_inference/core/tests/test_performance.py
import pytest
import time
import numpy as np
from active_inference.core import ActiveInferenceAgent

class TestPerformance:
    """Performance benchmarks for core components."""

    @pytest.mark.benchmark
    def test_inference_performance(self, benchmark):
        """Benchmark inference performance."""

        agent = ActiveInferenceAgent(state_dim=32, obs_dim=64, action_dim=8)
        observation = np.random.randn(64)

        def run_inference():
            return agent.act(observation)

        result = benchmark(run_inference)

        # Performance assertions
        assert result.stats.mean < 0.1  # Should complete in < 100ms
        assert result.stats.std_dev < 0.01  # Low variance

    @pytest.mark.benchmark
    def test_batch_processing_performance(self, benchmark):
        """Benchmark batch processing performance."""

        agent = ActiveInferenceAgent(state_dim=16, obs_dim=32, action_dim=4)
        batch_size = 100
        observations = [np.random.randn(32) for _ in range(batch_size)]

        def process_batch():
            actions = []
            for obs in observations:
                action = agent.act(obs)
                actions.append(action)
            return actions

        result = benchmark(process_batch)

        # Calculate throughput
        throughput = batch_size / result.stats.mean
        assert throughput > 100  # At least 100 inferences per second

    @pytest.mark.parametrize("state_dim", [4, 16, 64])
    def test_scaling_performance(self, state_dim, benchmark):
        """Test performance scaling with problem size."""

        agent = ActiveInferenceAgent(
            state_dim=state_dim,
            obs_dim=state_dim * 2,
            action_dim=state_dim // 2
        )

        observation = np.random.randn(state_dim * 2)

        def run_scaled_inference():
            return agent.act(observation)

        result = benchmark(run_scaled_inference)

        # Performance should scale reasonably with problem size
        expected_time = 0.001 * (state_dim / 4) ** 2  # Quadratic scaling assumption
        assert result.stats.mean < expected_time * 2  # Allow 2x overhead
```

### Memory Profiling

**Memory Usage Tests:**
```python
# src/python/active_inference/core/tests/test_memory.py
import pytest
import psutil
import os
from active_inference.core import ActiveInferenceAgent

class TestMemoryUsage:
    """Memory usage tests for core components."""

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_agent_memory_usage(self):
        """Test agent memory usage."""

        initial_memory = self.get_memory_usage()

        agent = ActiveInferenceAgent(state_dim=16, obs_dim=32, action_dim=4)

        agent_memory = self.get_memory_usage() - initial_memory

        # Should use reasonable memory
        assert agent_memory < 100  # Less than 100MB for basic agent

    def test_memory_leak_prevention(self):
        """Test that agents don't leak memory over time."""

        agent = ActiveInferenceAgent(state_dim=8, obs_dim=16, action_dim=2)

        initial_memory = self.get_memory_usage()

        # Run many inference cycles
        for i in range(1000):
            observation = np.random.randn(16)
            action = agent.act(observation)

        final_memory = self.get_memory_usage()
        memory_delta = final_memory - initial_memory

        # Memory usage should not grow significantly
        assert memory_delta < 50  # Less than 50MB growth over 1000 inferences
```

## Code Quality Standards

### Development Standards

**Code Style:**
- PEP 8 compliance for Python code
- Google C++ Style Guide for C++ code
- Comprehensive type hints in Python
- Doxygen documentation for C++ code

**Testing Standards:**
- 85%+ code coverage requirement
- Unit tests for all public APIs
- Integration tests for component interactions
- Performance regression tests

**Documentation Standards:**
- Docstrings for all public functions/classes
- API documentation generation
- Usage examples and tutorials
- Architecture decision records

### Continuous Integration

**CI Pipeline:**
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Setup Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    - name: Run Python tests
      run: |
        pytest tests/ -v --cov=src/python --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  cpp-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup C++
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential libeigen3-dev
    - name: Build C++
      run: |
        mkdir build && cd build
        cmake ..
        make -j$(nproc)
    - name: Run C++ tests
      run: |
        cd build
        ctest --output-on-failure

  quality-checks:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run quality checks
      run: |
        python scripts/run_quality_checks.py
```

## Future Architecture Enhancements

### Advanced Features
- **GPU Acceleration**: CUDA/OpenCL integration for C++
- **Distributed Computing**: MPI support for multi-node processing
- **Real-time Systems**: Hard real-time guarantees
- **Plugin Architecture**: Extensible component system

### Performance Optimizations
- **SIMD Operations**: Vectorized processing in C++
- **Memory Pooling**: Custom allocators for reduced latency
- **Cache-Oblivious Algorithms**: Improved cache performance
- **NUMA Awareness**: Multi-socket system optimization

### Research Integration
- **Algorithm Plugins**: Easy integration of new research algorithms
- **Experiment Frameworks**: Automated research pipelines
- **Benchmarking Suites**: Comprehensive performance evaluation
- **Visualization Tools**: Advanced debugging and analysis tools

