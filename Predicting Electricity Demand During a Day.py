import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# 1. Load CSV data
# ---------------------------------------------------

df = pd.read_csv("/content/electricity_demand_large.csv")

# Input and target
X = df[['time_normalized']].values
y = df['measured_demand'].values
y_true = df['true_demand'].values

# ---------------------------------------------------
# 2. Train / Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
    X, y, y_true, test_size=0.25, random_state=42
)

# ---------------------------------------------------
# 3. Model capacity (number of RFF components)
# ---------------------------------------------------

n_components_list = [5, 10, 25, 50, 100, 200]

train_errors = []
test_errors = []

# ---------------------------------------------------
# 4. Plot predictions for different capacities
# ---------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, n_components in enumerate(n_components_list):
    # Random Fourier Features
    rff = RBFSampler(
        n_components=n_components,
        gamma=1.0,
        random_state=42
    )

    X_train_rff = rff.fit_transform(X_train)
    X_test_rff = rff.transform(X_test)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train_rff, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_rff)
    y_test_pred = model.predict(X_test_rff)

    # Errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

    # Sort for smooth plotting
    sort_idx = np.argsort(X_test.ravel())

    ax = axes[idx]
    ax.scatter(
        X_train, y_train,
        c="red", s=20, alpha=0.5,
        label="Training data"
    )
    ax.plot(
        X_test[sort_idx],
        y_true_test[sort_idx],
        "b-", linewidth=2,
        label="True demand"
    )
    ax.plot(
        X_test[sort_idx],
        y_test_pred[sort_idx],
        "g--", linewidth=2,
        label="Prediction"
    )

    ax.set_title(
        f"RFF Components: {n_components}\n"
        f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}"
    )
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Electricity Demand")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("electricity_rff_predictions.png", dpi=150)
print("Saved: electricity_rff_predictions.png")

# ---------------------------------------------------
# 5. Error vs Model Capacity
# ---------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(
    n_components_list, train_errors,
    "o-", linewidth=2, label="Training MSE"
)
plt.plot(
    n_components_list, test_errors,
    "s-", linewidth=2, label="Test MSE"
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of RFF Components")
plt.ylabel("Mean Squared Error")
plt.title("Electricity Demand Prediction\nError vs Model Capacity")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("electricity_rff_error_curve.png", dpi=150)
print("Saved: electricity_rff_error_curve.png")

# ---------------------------------------------------
# 6. Overfitting demonstration
# ---------------------------------------------------

n_overfit = 500

rff_overfit = RBFSampler(
    n_components=n_overfit,
    gamma=1.0,
    random_state=42
)

X_train_overfit = rff_overfit.fit_transform(X_train)
X_test_overfit = rff_overfit.transform(X_test)

model_overfit = LinearRegression()
model_overfit.fit(X_train_overfit, y_train)

y_train_overfit = model_overfit.predict(X_train_overfit)
y_test_overfit = model_overfit.predict(X_test_overfit)

plt.figure(figsize=(8, 6))

sort_idx = np.argsort(X_test.ravel())

plt.scatter(
    X_train, y_train,
    c="red", s=25, alpha=0.6,
    label="Training data"
)
plt.plot(
    X_test[sort_idx],
    y_true_test[sort_idx],
    "b-", linewidth=2.5,
    label="True demand"
)
plt.plot(
    X_test[sort_idx],
    y_test_overfit[sort_idx],
    "g--", linewidth=2,
    label=f"Prediction (n={n_overfit})"
)

plt.title(
    f"Overfitting Example\n"
    f"Train MSE: {mean_squared_error(y_train, y_train_overfit):.4f}, "
    f"Test MSE: {mean_squared_error(y_test, y_test_overfit):.4f}"
)
plt.xlabel("Time (normalized)")
plt.ylabel("Electricity Demand")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("electricity_rff_overfitting.png", dpi=150)
print("Saved: electricity_rff_overfitting.png")

# ---------------------------------------------------
# 7. Summary Table
# ---------------------------------------------------

print("\n" + "=" * 60)
print("ELECTRICITY DEMAND – UNIVERSAL APPROXIMATION DEMO")
print("=" * 60)
print(f"{'Components':<12} {'Train MSE':<15} {'Test MSE':<15} {'Ratio':<10}")
print("-" * 60)

for n, tr, te in zip(n_components_list, train_errors, test_errors):
    ratio = te / tr if tr > 0 else np.inf
    print(f"{n:<12} {tr:<15.6f} {te:<15.6f} {ratio:<10.2f}")

print("-" * 60)
print("✓ Increasing capacity reduces training error")
print("✓ Moderate capacity generalizes best")
print("✓ Very high capacity leads to overfitting")
print("=" * 60)

plt.show()
