import pandas as pd
import matplotlib.pyplot as plt

# Load results from run_experiments
df = pd.read_csv("results/results_raw/experiment_outputs.csv")

# Compute grounded accuracy
accuracy = 1 - df["final_hallucination"].mean()

print(f"Grounded Answer Accuracy: {accuracy:.2%}")

# Plot
plt.figure(figsize=(6,5))
plt.bar(["Accuracy"], [accuracy], color="green")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Grounded Answer Accuracy")
plt.text(0, accuracy/2, f"{accuracy:.1%}", ha="center", va="center", fontsize=14)
plt.tight_layout()
plt.savefig("results/plots/grounded_accuracy.png")
plt.show()
