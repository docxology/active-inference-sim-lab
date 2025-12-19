# Documentation Architecture - AGENTS

## Module Overview

The `docs` directory contains comprehensive technical documentation for the Active Inference Simulation Lab, providing architectural overviews, API references, tutorials, and deployment guides.

## Documentation Structure

```
docs/
├── AGENTS.md              # This file - Documentation architecture
├── ARCHITECTURE.md        # System architecture overview
├── API_REFERENCE.md       # Complete API documentation
├── DEPLOYMENT_GUIDE.md    # Production deployment instructions
├── DEVELOPMENT_GUIDE.md   # Developer onboarding and contribution
├── PERFORMANCE_GUIDE.md   # Performance optimization and benchmarking
├── SECURITY_GUIDE.md      # Security best practices and hardening
├── TROUBLESHOOTING.md     # Common issues and solutions
├── CHANGELOG.md           # Version history and release notes
├── CONTRIBUTING.md        # Contribution guidelines
├── CODE_OF_CONDUCT.md     # Community standards
├── LICENSE                # Project licensing
├── api/                   # API documentation by component
├── tutorials/            # Step-by-step tutorials
├── examples/             # Code examples and use cases
├── images/               # Diagrams and screenshots
└── archive/              # Historical documentation
```

## Documentation Architecture

### Content Organization

**Technical Documentation:**
- **Architecture**: System design, component interactions, data flows
- **API Reference**: Complete API documentation with examples
- **Deployment**: Production deployment and configuration
- **Development**: Setup, contribution, and development workflows

**User Documentation:**
- **Tutorials**: Step-by-step guides for common tasks
- **Examples**: Practical code examples and use cases
- **Troubleshooting**: Common issues and solutions
- **Performance**: Optimization guides and benchmarks

**Community Documentation:**
- **Contributing**: How to contribute to the project
- **Code of Conduct**: Community standards and guidelines
- **Changelog**: Release notes and version history

## Documentation Standards

### Content Guidelines

**Structure and Format:**
- Use Markdown for all documentation
- Follow consistent heading hierarchy (H1 → H6)
- Include table of contents for documents >3 sections
- Use code blocks with syntax highlighting
- Include cross-references to related documentation

**Style Guidelines:**
- Write in active voice, present tense
- Use "you" for user-facing instructions
- Be concise but comprehensive
- Include practical examples
- Use consistent terminology

**Quality Standards:**
- Technical accuracy verified by subject matter experts
- Peer review for all major documentation updates
- Regular updates to maintain currency
- Accessibility considerations (alt text, semantic markup)

### Documentation Types

**Reference Documentation:**
- API specifications and method signatures
- Configuration parameters and options
- Data structures and schemas
- Error codes and messages

**Procedural Documentation:**
- Installation and setup instructions
- Configuration and deployment guides
- Troubleshooting and maintenance procedures
- Upgrade and migration guides

**Conceptual Documentation:**
- Architecture and design overviews
- Theory and background information
- Best practices and recommendations
- Performance characteristics and limitations

## API Documentation

### API Reference Structure

**Endpoint Documentation:**
```markdown
### GET /api/v1/agents/{agent_id}

Retrieve information about a specific active inference agent.

**Parameters:**
- `agent_id` (path): Unique identifier for the agent

**Response:**
```json
{
  "agent_id": "string",
  "status": "active|inactive",
  "configuration": {
    "state_dim": 4,
    "obs_dim": 8,
    "action_dim": 2
  },
  "performance_metrics": {
    "inference_time_ms": 45.2,
    "memory_usage_mb": 128.5,
    "cache_hit_ratio": 0.87
  }
}
```

**Error Responses:**
- `404 Not Found`: Agent not found
- `500 Internal Server Error`: Server error

**Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/agents/agent_001" \\
  -H "Authorization: Bearer <token>"
```
```

**Class Documentation:**
```python
class ActiveInferenceAgent:
    """Primary agent implementation using active inference.

    This class provides the main interface for creating and managing
    active inference agents that learn through minimizing free energy.

    Attributes:
        agent_id (str): Unique identifier for the agent
        state_dim (int): Dimensionality of hidden state space
        obs_dim (int): Dimensionality of observation space
        action_dim (int): Dimensionality of action space

    Example:
        >>> agent = ActiveInferenceAgent(
        ...     state_dim=4,
        ...     obs_dim=8,
        ...     action_dim=2
        ... )
        >>> action = agent.act(observation)
    """

    def __init__(self, state_dim, obs_dim, action_dim, **kwargs):
        """Initialize the active inference agent.

        Args:
            state_dim (int): Hidden state dimensionality
            obs_dim (int): Observation dimensionality
            action_dim (int): Action dimensionality
            **kwargs: Additional configuration parameters
        """
```

## Tutorial Framework

### Tutorial Structure

**Progressive Learning Path:**
1. **Introduction**: Basic concepts and setup
2. **Foundations**: Core active inference principles
3. **Implementation**: Building basic agents
4. **Advanced Topics**: Research and extensions
5. **Production**: Deployment and scaling

**Tutorial Template:**
```markdown
# Tutorial: Building Your First Active Inference Agent

## Overview

In this tutorial, you'll learn how to create and train a basic active inference agent using the Active Inference Simulation Lab.

## Prerequisites

- Python 3.9+
- Basic understanding of reinforcement learning
- Familiarity with numpy and PyTorch

## Learning Objectives

By the end of this tutorial, you will be able to:
- Set up the development environment
- Create a simple active inference agent
- Train the agent on a basic task
- Analyze the agent's performance

## Step 1: Environment Setup

First, install the required dependencies:

```bash
pip install active-inference torch numpy matplotlib
```

## Step 2: Creating the Agent

```python
from active_inference import ActiveInferenceAgent
import numpy as np

# Create a simple agent
agent = ActiveInferenceAgent(
    state_dim=4,      # Hidden state dimensionality
    obs_dim=8,        # Observation dimensionality
    action_dim=2      # Action dimensionality
)
```

## Step 3: Training Loop

```python
# Training parameters
num_episodes = 1000
max_steps = 200

for episode in range(num_episodes):
    observation = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Get action from agent
        action = agent.act(observation)

        # Execute action in environment
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward

        # Update agent (if learning is enabled)
        if agent.learning_enabled:
            agent.update(observation, action, reward, next_obs)

        observation = next_obs

        if done:
            break

    print(f"Episode {episode}: Reward = {episode_reward}")
```

## Step 4: Analysis and Visualization

```python
import matplotlib.pyplot as plt

# Plot training progress
plt.plot(rewards_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()

# Analyze agent beliefs
beliefs = agent.get_beliefs()
print(f"Final beliefs: {beliefs}")
```

## Next Steps

- Experiment with different agent configurations
- Try more complex environments
- Explore advanced active inference features
- Deploy your agent to production

## Additional Resources

- [API Reference](api/agent.md)
- [Advanced Tutorials](tutorials/advanced.md)
- [Community Forum](https://forum.active-inference.org)
```

## Example Documentation

### Code Examples Repository

**Example Categories:**
- **Basic Usage**: Simple agent creation and training
- **Research Applications**: Novel algorithms and extensions
- **Production Deployment**: Real-world deployment scenarios
- **Integration Examples**: Third-party system integration

**Example Structure:**
```python
"""
Active Inference Grid World Example

This example demonstrates how to create an active inference agent
that learns to navigate a simple grid world environment.

The agent uses variational inference to update its beliefs about
the environment state and plans actions by minimizing expected
free energy.
"""

import numpy as np
from active_inference import ActiveInferenceAgent
from active_inference.environments import GridWorld

def main():
    """Main training and evaluation loop."""

    # Create environment
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
        action_dim=env.action_space.shape[0],
        inference_method='variational',
        planning_horizon=5
    )

    # Training loop
    num_episodes = 1000
    rewards_history = []

    print("Starting training...")

    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Agent selects action
            action = agent.act(observation)

            # Environment responds
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward

            # Agent learns from experience
            agent.update(observation, action, reward, next_obs)

            observation = next_obs

        rewards_history.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode}: Average reward = {avg_reward:.2f}")

    # Evaluation
    print("\\nEvaluating trained agent...")
    evaluate_agent(agent, env, num_episodes=100)

def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate trained agent performance."""

    total_rewards = []

    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(observation)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            observation = next_obs

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"Evaluation Results:")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Success rate: {np.mean([r > 0 for r in total_rewards]):.2%}")

if __name__ == "__main__":
    main()
```

## Documentation Maintenance

### Update Procedures

**Version Updates:**
- Update API documentation with each release
- Review and update tutorials for breaking changes
- Archive outdated examples and tutorials
- Update compatibility matrices

**Review Process:**
- Technical review by subject matter experts
- Editorial review for clarity and consistency
- User testing for tutorial accuracy
- Automated checks for broken links and formatting

### Quality Assurance

**Automated Checks:**
- Link validation and broken reference detection
- Formatting and style consistency checks
- Technical accuracy validation
- Accessibility compliance testing

**Manual Review:**
- Peer review for technical content
- User experience testing for tutorials
- Cross-platform compatibility verification
- Performance benchmark validation

## Integration with Development

### Documentation as Code

**Version Control:**
- Documentation stored alongside code
- Versioned with releases
- Branch-specific documentation for features
- Automated deployment of documentation

**Continuous Integration:**
```yaml
# .github/workflows/docs.yml
name: Documentation
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r docs/requirements.txt
      - name: Build documentation
        run: |
          mkdocs build
      - name: Check links
        run: |
          linkchecker _site/index.html
      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: _site
```

### API Documentation Generation

**Automated API Docs:**
```python
# docs/generate_api_docs.py
import inspect
import importlib
from typing import get_type_hints

def generate_api_docs(module_path, output_path):
    """Generate API documentation from code."""

    module = importlib.import_module(module_path)

    with open(output_path, 'w') as f:
        f.write("# API Reference\\n\\n")

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and not name.startswith('_'):
                f.write(f"## Class: {name}\\n\\n")

                # Class docstring
                if obj.__doc__:
                    f.write(f"{obj.__doc__}\\n\\n")

                # Methods
                for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if not method_name.startswith('_'):
                        f.write(f"### Method: {method_name}\\n\\n")

                        # Method signature
                        sig = inspect.signature(method)
                        f.write(f"```python\\n{method_name}{sig}\\n```\\n\\n")

                        # Method docstring
                        if method.__doc__:
                            f.write(f"{method.__doc__}\\n\\n")

                        # Type hints
                        hints = get_type_hints(method)
                        if hints:
                            f.write("**Type Hints:**\\n")
                            for param, hint in hints.items():
                                f.write(f"- `{param}`: `{hint}`\\n")
                            f.write("\\n")

if __name__ == "__main__":
    generate_api_docs("active_inference.core", "docs/api/core.md")
    generate_api_docs("active_inference.environments", "docs/api/environments.md")
```

## Future Documentation Enhancements

### Interactive Documentation
- **Jupyter Notebooks**: Executable tutorials and examples
- **Interactive Demos**: Web-based interactive demonstrations
- **API Explorer**: Interactive API documentation
- **Code Playgrounds**: Online coding environments

### Advanced Features
- **Video Tutorials**: Screencast tutorials for complex topics
- **Interactive Diagrams**: Dynamic architecture visualizations
- **Search and Discovery**: Advanced documentation search
- **Personalization**: User-specific documentation recommendations

### Collaboration Features
- **Community Contributions**: Crowdsourced documentation improvements
- **Translation Support**: Multi-language documentation
- **Feedback Integration**: User feedback and improvement tracking
- **Version Comparison**: Documentation diffing between versions

