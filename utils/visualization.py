from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch

# def plot_samples(dataset:MNIST, num_samples:int=6):
#     fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
#     for i in range(num_samples):
#         image, label = dataset[i]
#         # Denormalisieren für Visualisierung
#         image = image * 0.3081 + 0.1307
#         axes[i].imshow(image.squeeze(), cmap='gray')
#         axes[i].set_title(f'Label: {label}')
#         axes[i].axis('off')
#     plt.tight_layout()
#     plt.show()

def plot_series(ylabel:str, train_values:list, val_values:list):
    plt.figure(figsize=(8, 5))
    plt.plot(train_values, label=f'train')
    plt.plot(val_values, label=f'val')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.minorticks_on()
    plt.xlim(0, len(train_values)-1)
    plt.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
    plt.legend()
    plt.show()

def plot_samples(X:torch.Tensor, Y:torch.Tensor, Y_hat:torch.Tensor = None, n_plots:int = 10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(n_plots):
        image, true_label = X[i], Y[i]

        # Denormalisieren für Visualisierung
        image_display = image * 0.3081 + 0.1307
        
        # Plot
        axes[i].imshow(image_display.squeeze(), cmap='gray')

        if Y_hat is not None:
            image_tensor = Y_hat[i].unsqueeze(0)  # Form anpassen für das Modell (Batch-Größe von 1)
            predicted_label = torch.argmax(image_tensor, dim=1).item()
            color = 'green' if predicted_label == true_label else 'red'
            axes[i].set_title(f'Pred: {predicted_label}, True: {true_label}', 
                            color=color, fontsize=12, fontweight='bold')
        else:
            axes[i].set_title(f'True: {true_label}', fontsize=12, fontweight='bold')

        axes[i].axis('off')

    plt.tight_layout()
    plt.show()