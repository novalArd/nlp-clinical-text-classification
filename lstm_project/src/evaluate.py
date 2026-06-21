import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, label_names=None):
    """Evaluasi model pada data test."""
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(1).numpy()
    
    y_true = y_test.numpy()
    
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average='weighted', zero_division=0)
    rec  = recall_score(y_true, preds, average='weighted', zero_division=0)
    f1   = f1_score(y_true, preds, average='weighted', zero_division=0)
    
    return acc, prec, rec, f1
