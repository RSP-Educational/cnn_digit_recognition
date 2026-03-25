import torch
import torch.nn as nn

def accuracy(preds, targets):
    """
    Berechnet die Genauigkeit (Accuracy) zwischen den Vorhersagen und den tatsächlichen Labels.
    
    Args:
        preds (torch.Tensor): Vorhersagen des Modells (z.B. Logits oder Softmax-Ausgaben).
        targets (torch.Tensor): Tatsächliche Labels (z.B. Integer-Klassen).
    
    Returns:
        float: Genauigkeit als Wert zwischen 0 und 1.
    """
    # 1. Vorhersagen in Klassen umwandeln (z.B. durch argmax)
    predicted_classes = torch.argmax(preds, dim=1)
    
    # 2. Anzahl der korrekten Vorhersagen zählen
    correct_predictions = (predicted_classes == targets).sum().item()
    
    # 3. Gesamtzahl der Beispiele
    total_predictions = targets.size(0)
    
    # 4. Genauigkeit berechnen
    accuracy_value = correct_predictions / total_predictions
    
    return accuracy_value

class CCELoss(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(CCELoss, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.epsilon = epsilon  # Verhindert log(0) Probleme

    def forward(self, softmax_outputs, targets):
        # 1. Numerische Stabilität: Werte leicht einschränken
        softmax_outputs = torch.clamp(softmax_outputs, self.epsilon, 1.0 - self.epsilon)
        
        # 2. Logarithmus berechnen (macht aus Softmax -> LogSoftmax)
        log_probs = torch.log(softmax_outputs)
        
        # 3. NLLLoss anwenden
        return self.nll_loss(log_probs, targets)