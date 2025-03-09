# Double Pendulum Chaos Analysis

## Project Overview

This project simulates the dynamics of a double pendulum, a well-known example of chaotic motion. By numerically solving the equations of motion, analyzing energy conservation, and visualizing phase space, we gain insights into the system's behavior. Additionally, we compute Lyapunov exponents and approximate entropy to quantify chaos and train a machine learning model to predict chaotic behavior from initial conditions.

## Accomplishments

- Implemented a numerical solver for the double pendulum using `scipy.integrate.solve_ivp`.
- Visualized motion using Matplotlib animations.
- Analyzed energy conservation to verify the correctness of the simulation.
- Computed Lyapunov exponents to measure sensitivity to initial conditions.
- Estimated entropy to evaluate system complexity.
- Trained a neural network to predict chaotic behavior from initial conditions.
- Developed analytical plots to visualize relationships between chaos indicators.

## How It Works

### Numerical Simulation

- The equations of motion are solved using a system of coupled differential equations.
- The state variables (angles & angular velocities) are evolved over time.

### Energy Analysis

- Computes kinetic & potential energy to verify total energy conservation.

### Chaos Quantification

#### Lyapunov exponent
- Measures the rate at which trajectories diverge.

#### Approximate entropy
- Evaluates the complexity of time series data.

### Machine Learning Model

- Trains a neural network using TensorFlow/Keras to predict Lyapunov exponents.
- Uses initial conditions as input features.

### Visualization

- Animates the double pendulum motion.
- Plots phase space trajectories.
- Displays relationships between entropy, energy, and chaos.

## Why It’s Important

- **Demonstrates Chaos Theory**: The double pendulum is a classic example of a chaotic system.
- **Quantifies Unpredictability**: Computing Lyapunov exponents helps quantify unpredictability in real-world systems like weather forecasting and financial markets.
- **Machine Learning Application**: The ML model showcases how AI can help predict complex system behavior.
- **Educational Value**: Serves as a learning resource for nonlinear dynamics, numerical methods, and deep learning.

## Installation & Usage

### Prerequisites

Ensure you have Python 3 installed along with the following dependencies:

```bash
pip install numpy scipy matplotlib tensorflow keras
