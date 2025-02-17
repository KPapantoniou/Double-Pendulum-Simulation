import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from simulation import run_simulation
from scipy.spatial.distance import euclidean
import keras
from keras import layers


m1_values = [1] ##[1.0, 2.0]
m2_values = [1] ##[1.0, 2.0]
L1_values = [1] ##[0.5, 1.0]
L2_values = [1] ##[0.5, 1.0]
theta_initial_range = np.linspace(0, 180, 5) 
omega_initial_range = np.linspace(-1, 1, 5)  

g = 9.81

results = []

def compute_lyapunov(m1,m2,L1,L2, theta1_init, omega1_init, theta2_init,omega2_init, epsilon = 1e-5):
    """Compute the largest Lyapunov exponent by measuring divergence of two close trajectories."""
    time_eval,theta1, theta2, *_ = run_simulation(m1,m2,L1,L2,g,theta1_init,omega1_init,theta2_init,omega2_init) 
    _, theta1_pertubate, theta2_pertubate, *_ = run_simulation(m1,m2,L1,L2,g,theta1_init + epsilon,omega1_init,theta2_init,omega2_init) 
    d= np.abs(theta1_pertubate - theta1) + np.abs(theta2_pertubate-theta2)
    lyap_exp = np.polyfit(time_eval, np.log(d+1e-12),1)[0]
    return lyap_exp    

def approximate_entropy(U, m=2, r=0.2):
    """Compute Approximate Entropy (ApEn) for a time series U."""
    def _phi(m):
        patterns = np.array([U[i:i + m] for i in range(len(U) - m + 1)])
        C = np.mean([np.sum(np.max(np.abs(patterns - p), axis=1) <= r) / len(patterns) for p in patterns])
        return np.log(C)
    return np.abs(_phi(m) - _phi(m + 1))
    
for m1 in m1_values:
    for m2 in m2_values:
        for L1 in L1_values:  
            for L2 in L2_values:
                for theta1_initial in theta_initial_range:
                    for theta2_initial in theta_initial_range:
                        for omega1_initial in omega_initial_range:
                            for omega2_initial in omega_initial_range:                   
                                time_eval, theta1, theta2, KE, PE, total_energy, x1, y1, x2, y2, omega1, omega2 = run_simulation(m1, m2, L1, L2, g, theta1_initial, omega1_initial, theta2_initial, omega2_initial)
                                
                                lyap_exp = compute_lyapunov(m1,m2,L1,L2,theta1_initial,omega1_initial,theta2_initial,omega2_initial)
                                
                                entropy = approximate_entropy(theta1)
                                                                                            
                                results.append({
                                    'theta1_initial': theta1_initial,
                                    'theta2_initial': theta2_initial,
                                    'omega1_initial': omega1_initial,
                                    'omega2_initial': omega2_initial,
                                    'lyapunov_exp': lyap_exp,
                                    'entropy': entropy,
                                    'energy_variation': np.ptp(total_energy)
                                })


def tf_model():
    model = keras.Sequential([
        layers.Dense(64,activation='relu', input_shape=(4,)),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse', metrics = ['mae'])
    return model

def train_model(results):
    X = np.array([[r['theta1_initial'],r['theta2_initial'],r['omega1_initial'],r['omega2_initial']]for r in results])
    Y = np.array([r['lyapunov_exp']]for r in results)
    
    model = tf_model()
    model.fit(X,Y, epochs=100, verbose=1, batch_size=8)
    return model

ml_model = train_model(results)

def predict_lyapunov(theta1,theta2,omega1,omega2):
    return ml_model.predict(np.array([[theta1,theta2,omega1,omega2]]))[0][0]

for result in results:

    theta1_initial = result['theta1_initial']
    theta2_initial = result['theta2_initial']
    omega1_initial = result['omega1_initial']
    omega2_initial = result['omega2_initial']
    

    predicted_lyap = predict_lyapunov(theta1_initial, theta2_initial, omega1_initial, omega2_initial)
    

    result['predicted_lyap'] = predicted_lyap

    print(f"Predicted Lyapunov Exponent for initial conditions (theta1: {theta1_initial}, theta2: {theta2_initial}): {predicted_lyap}")

def plot_analysis(results):
    # Actual vs Predicted Lyapunov
    plt.figure(figsize=(10, 6))
    # y_pred = model.predict(scaler.transform(np.array([
    #     [r['theta1_initial'], r['theta2_initial'], 
    #      r['omega1_initial'], r['omega2_initial']] for r in results]
    # )))
    plt.scatter([r['lyapunov_exp'] for r in results], result['predicted_lyap'], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Lyapunov Exponent')
    plt.ylabel('Predicted Lyapunov Exponent')
    plt.title('Chaos Prediction Accuracy')
    plt.grid()
    
    # Entropy vs Lyapunov
    plt.figure(figsize=(10, 6))
    plt.scatter([r['entropy'] for r in results], [r['lyapunov_exp'] for r in results], 
                c=[r['energy_variation'] for r in results], cmap='viridis')
    plt.colorbar(label='Energy Variation')
    plt.xlabel('Approximate Entropy')
    plt.ylabel('Lyapunov Exponent')
    plt.title('Entropy vs Chaos Intensity')

plot_analysis(results)

        