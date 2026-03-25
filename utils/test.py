from enum import Enum
import torch
from torch.utils.data import DataLoader
import utils.visualization as vis


class TaskType(Enum):
    REGRESSION = 0
    MULTI_CLASS_CLASSIFICATION = 1
    MULTI_LABEL_CLASSIFICATION = 2
    GENERATION = 3

def test_task_type(taskType:TaskType):
    if taskType == TaskType.REGRESSION:
        print("❌ Regression Tasks haben kontinuierliche Zielwerte. Das ist hier nicht der Fall.")
    elif taskType == TaskType.MULTI_CLASS_CLASSIFICATION:
        print("✅ Da pro Bild nur genau eine Klasse korrekt sein kann, handelt es sich um eine Multi-Class Classification.")
    elif taskType == TaskType.MULTI_LABEL_CLASSIFICATION:
        print("❌ Multi-Label Tasks erlauben mehrere korrekte Klassen pro Bild. Hier kommt genau eine Klasse pro Bild vor.")
    elif taskType == TaskType.GENERATION:
        print("❌ Generative Tasks beinhalten die Erstellung neuer Daten (z.B. Bilder, Texte). Hier geht es um Klassifikation.")
    else:
        print("Unknown Task Type")

class ActivationType(Enum):
    RELU = 0
    SIGMOID = 1
    TANH = 2
    SOFTMAX = 3

def test_activation_type(activationType:ActivationType):
    if activationType == ActivationType.RELU:
        print("❌ ReLU ist eine häufige Aktivierungsfunktion, aber für die Ausgabe von Multi-Class Classification Modellen wird sie nicht verwendet.")
    elif activationType == ActivationType.SIGMOID:
        print("❌ Sigmoid wird oft für Multi-Label Classification verwendet, da es Werte zwischen 0 und 1 für jede Klasse ausgibt. Hier haben wir jedoch eine Multi-Class Classification.")
    elif activationType == ActivationType.TANH:
        print("❌ Tanh ist eine Aktivierungsfunktion, die Werte zwischen -1 und 1 ausgibt. Sie konvertiert die Ausgaben also nicht in Wahrscheinlichkeiten.")
    elif activationType == ActivationType.SOFTMAX:
        print("✅ Softmax ist die richtige Wahl für Multi-Class Classification, da sie die Ausgaben in Wahrscheinlichkeiten umwandelt, die sich auf 1 summieren.\n" \
        "   Außerdem führt Softmax eine Klassenkonkurrenz ein, was hohe Werte verstärkt und niedrige Werte weiter abschwächt.")
    else:
        print("Unknown Activation Type")

class LossType(Enum):
    MSE = 0
    BCE = 1
    CCE = 2
    MAE = 3

def test_loss_type(lossType:LossType):
    if lossType == LossType.MSE:
        print("❌ MSE (Mean Squared Error) ist eine Verlustfunktion, die hauptsächlich für Regressionsaufgaben verwendet wird. Sie misst den durchschnittlichen quadratischen\n" \
        "   Fehler zwischen den vorhergesagten und tatsächlichen Werten.")
    elif lossType == LossType.BCE:
        print("❌ BCE (Binary Cross-Entropy) ist eine Verlustfunktion, die für binäre Klassifikationsaufgaben oder Multi-Label Classification verwendet wird. Sie berechnet die\n" \
        "   Kreuzentropie zwischen den vorhergesagten Wahrscheinlichkeiten und den tatsächlichen Labels.")
    elif lossType == LossType.CCE:
        print("✅ CCE (Categorical Cross-Entropy) ist die richtige Wahl für Multi-Class Classification, da sie die Kreuzentropie zwischen den vorhergesagten Wahrscheinlichkeiten\n"
        "   (nach Softmax) und den tatsächlichen Klassenlabels berechnet.")
    elif lossType == LossType.MAE:
        print("❌ MAE (Mean Absolute Error) ist eine Verlustfunktion, die hauptsächlich für Regressionsaufgaben verwendet wird. Sie misst den durchschnittlichen\n" \
        "   absoluten Fehler zwischen den vorhergesagten und tatsächlichen Werten.")
    else:
        print("Unknown Loss Type")

def test_model_pytorch(model:torch.nn.Module, dataloader:DataLoader, device:torch.device):
    model.eval()
    correct = 0
    total = 0

    X_correct, Y_correct, Y_hat_correct = [], [], []
    X_incorrect, Y_incorrect, Y_hat_incorrect = [], [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            # Collect correct and incorrect predictions
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    X_correct.append(inputs[i].cpu())
                    Y_correct.append(labels[i].cpu())
                    Y_hat_correct.append(outputs[i].cpu())
                else:
                    X_incorrect.append(inputs[i].cpu())
                    Y_incorrect.append(labels[i].cpu())
                    Y_hat_incorrect.append(outputs[i].cpu())

    X = X_correct[:5] + X_incorrect[:5]
    Y = Y_correct[:5] + Y_incorrect[:5]
    Y_hat = Y_hat_correct[:5] + Y_hat_incorrect[:5]
    vis.plot_samples(X, Y, Y_hat)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')