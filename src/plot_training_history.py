import os
import matplotlib.pyplot as plt

from config import OUTPUT_DIR

# Du lieu train da ghi lai
epochs = list(range(1, 11))

accuracy = [0.7617, 0.9165, 0.9415, 0.9526, 0.9634, 0.9640, 0.9718, 0.9737, 0.9756, 0.9756]
loss = [0.7332, 0.2638, 0.1859, 0.1462, 0.1182, 0.1061, 0.0874, 0.0785, 0.0733, 0.0679]

val_accuracy = [0.9362, 0.9619, 0.9730, 0.9782, 0.9816, 0.9795, 0.9834, 0.9834, 0.9896, 0.9872]
val_loss = [0.2036, 0.1275, 0.0852, 0.0705, 0.0585, 0.0650, 0.0520, 0.0466, 0.0355, 0.0400]

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, marker="o", label="Train Accuracy")
plt.plot(epochs, val_accuracy, marker="o", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "accuracy_loss_chart.png")
plt.savefig(save_path, dpi=300)
plt.close()

print("Da tao lai file:", save_path)