import json
import matplotlib.pyplot as plt
import pandas as pd

# Load the training history from CSV file
history = pd.read_csv('training_log.csv')

# Define metrics to plot (excluding learning_rate)
metrics = ['accuracy', 'loss', 'auc', 'top_3_accuracy']

# Create subplots
plt.figure(figsize=(20, 5))

for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 4, i)
    
    # Plot training metric
    plt.plot(history[metric], label=f'Training {metric}')
    
    # Plot validation metric
    val_metric = f'val_{metric}'
    if val_metric in history.columns:
        plt.plot(history[val_metric], label=f'Validation {metric}')
    
    plt.title(f'Model {metric.replace("_", " ").title()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Updated training_history.png has been generated with new training metrics!") 