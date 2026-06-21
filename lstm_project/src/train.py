import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import time

def train_model(model, X_tr, y_tr, X_te, y_te, lr=1e-3, epochs=25, batch_size=16, max_norm=1.0):
    """Loop training standar dengan gradient clipping."""
    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[]}
    start = time.time()
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct = 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            
            # Gradient clipping mencegah exploding gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct    += (out.argmax(1) == yb).sum().item()
            
        # Evaluasi di data test (tanpa gradient)
        model.eval()
        with torch.no_grad():
            test_out  = model(X_te)
            test_loss = criterion(test_out, y_te).item()
            test_acc  = (test_out.argmax(1)==y_te).float().mean().item()
            
        history['train_loss'].append(total_loss / len(X_tr))
        history['train_acc'].append(correct / len(X_tr))
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Train Acc: {history['train_acc'][-1]:.4f} | "
              f"Test Loss: {history['test_loss'][-1]:.4f} | "
              f"Test Acc: {history['test_acc'][-1]:.4f}")
              
    elapsed = time.time() - start
    return history, elapsed
