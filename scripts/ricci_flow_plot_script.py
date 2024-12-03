import numpy as np
import matplotlib.pyplot as plt

# Number of points for A, W, and t
num_points = 500


# Function definitions
def compute_a_t(A, W, t):
    """
    Compute a(i,j)^t = Aij * e^t / (1 + Aij * W(i,j) * (e^t - 1)).
    """
    numerator = A * np.exp(t)
    denominator = 1 + A * W * (np.exp(t) - 1)
    return numerator / denominator


def compute_da_t(A, W, t):
    """
    Compute the derivative of a(i,j)^t with respect to t:
    da(i,j)^t/dt = |(Aij * e^t * (Aij * Wij - 1)) / (1 + Aij * Wij * (e^t - 1))^2|
    """
    numerator = A * np.exp(t) * (A * W - 1)
    denominator = (1 + A * W * (np.exp(t) - 1)) ** 2
    return np.abs(numerator / denominator)


# Generate random A and W values, ensuring half have Aij * W(i, j) > 1 and half < 1
np.random.seed(42)  # For reproducibility
t_values = np.linspace(0, 10, num_points)  # Range of t

# Splitting A and W into two groups based on Aij * Wij
A_high = np.random.uniform(0.5, 2, num_points // 2)
W_high = np.random.uniform(0.5, 2, num_points // 2)

A_low = np.random.uniform(0.1, 0.5, num_points // 2)
W_low = np.random.uniform(0.1, 0.5, num_points // 2)

# Combine into unified arrays
A_values = np.concatenate([A_high, A_low])
W_values = np.concatenate([W_high, W_low])

# Initialize matrices to store results
a_matrix = []
da_matrix = []
increasing_a_matrix = []
decreasing_a_matrix = []

# Compute a_t and da_t for all A, W pairs
plt.figure(figsize=(12, 8))

for i in range(num_points):
    A = A_values[i]
    W = W_values[i]

    # Compute a_t and da_t
    a_vector = compute_a_t(A, W, t_values)
    da_vector = compute_da_t(A, W, t_values)

    # Plot results
    plt.plot(t_values, a_vector, linestyle="dashed", color="gray", linewidth=0.5, alpha=0.5)
    plt.plot(t_values, da_vector, linewidth=0.7, alpha=0.5)

    # Store results
    a_matrix.append(a_vector)
    da_matrix.append(da_vector)

    if A * W < 1:
        increasing_a_matrix.append(a_vector)  # Aij * Wij < 1 (Increasing curves)
    else:
        decreasing_a_matrix.append(a_vector)  # Aij * Wij > 1 (Decreasing curves)

# Convert results to numpy arrays
a_matrix = np.array(a_matrix)
da_matrix = np.array(da_matrix)
increasing_a_matrix = np.array(increasing_a_matrix)
decreasing_a_matrix = np.array(decreasing_a_matrix)

# Find the maximum value of a_t at the final t
final_t_index = -1  # Last index of t_values
final_a_values = a_matrix[:, final_t_index]
max_index = np.argmax(final_a_values)
max_curve = a_matrix[max_index]

# Find the first t where max_curve > max(decreasing curves)
target_t = 0
for t_idx in range(num_points):
    max_decreasing_a = np.max(decreasing_a_matrix[:, t_idx])
    if max_curve[t_idx] > max_decreasing_a:
        target_t = t_idx
        break

highlight_start_t = t_values[target_t]

# Find the first t where max(a_t) > max(da_t)
target_t_2 = 0
for t_idx in range(num_points):
    max_a_t = np.max(a_matrix[:, t_idx])
    max_da_t = np.max(da_matrix[:, t_idx])
    if max_a_t > max_da_t:
        target_t_2 = t_idx
        break

highlight_end_t = t_values[target_t_2]

# Plot the max curve
plt.plot(t_values, max_curve, color="black", linewidth=2, label="Max Curve")

# Highlight the region between target_t_2 and target_t
plt.axvspan(highlight_start_t, highlight_end_t, color="yellow", alpha=0.3, label="Low Error Region")

# Plot formatting
plt.title("Network Change under Ricci Flow", fontsize=14)
plt.xlabel("$t$", fontsize=12)
plt.ylabel("$a(i,j)^t, |\\frac{d{a(i,j)}^t}{dt}|$", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.legend()

# Show plot
plt.show()
