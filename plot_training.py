import pandas as pd
import matplotlib.pyplot as plt

# Load training log
log_file = "logs/training_log.csv"
df = pd.read_csv(log_file)

print("Loaded log with", len(df), "entries")

# Plot Reward
plt.figure()
plt.plot(df["step"], df["reward"])
plt.title("Training Reward over Time")
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

# Plot Coordination Efficiency Score
plt.figure()
plt.plot(df["step"], df["ces"])
plt.title("Coordination Efficiency Score (CES)")
plt.xlabel("Training Steps")
plt.ylabel("CES")
plt.grid(True)
plt.show()

# Plot Value Loss
plt.figure()
plt.plot(df["step"], df["value_loss"])
plt.title("Value Loss during Training")
plt.xlabel("Training Steps")
plt.ylabel("Value Loss")
plt.grid(True)
plt.show()
