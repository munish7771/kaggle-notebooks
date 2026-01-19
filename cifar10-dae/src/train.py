import torch
import torch.nn as nn
import torch.optim as optim
import time
from .data import AddGaussianNoise

def train_classifier(model, loader, criterion, optimizer, device, epochs=10, log_interval=100):
    model.train()
    history = {'loss': [], 'time_per_epoch': []}
    
    total_steps = 0
    start_total_time = time.time()
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            # Map labels 3->0, 5->1 if necessary?
            # CIFAR labels are 0-9. Cats are 3, Dogs are 5.
            # Our classifier has output size 2. We need to map 3->0, 5->1.
            # Let's do it on the fly.
            target = torch.where(target == 3, torch.tensor(0, device=device), target)
            target = torch.where(target == 5, torch.tensor(1, device=device), target)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_steps += 1
            
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        epoch_time = time.time() - start_time
        history['time_per_epoch'].append(epoch_time)
        history['loss'].append(running_loss / len(loader))
        print(f'Result: Epoch {epoch} finished in {epoch_time:.2f}s, Avg Loss: {running_loss/len(loader):.4f}')
        
    total_time = time.time() - start_total_time
    history['total_time'] = total_time
    history['steps'] = total_steps
    return history

def train_dae(model, loader, criterion, optimizer, device, epochs=10, noise_std=0.1, log_interval=100):
    model.train()
    noise_adder = AddGaussianNoise(std=noise_std)
    history = {'loss': [], 'time_per_epoch': []}
    
    start_total_time = time.time()
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for batch_idx, (data, _) in enumerate(loader):
            # No labels needed for DAE (unsupervised)
            clean_data = data.to(device)
            noisy_data = noise_adder(data).to(device)
            
            optimizer.zero_grad()
            output = model(noisy_data)
            loss = criterion(output, clean_data) # Reconstruction loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % log_interval == 0:
                print(f'DAE Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

        epoch_time = time.time() - start_time
        history['time_per_epoch'].append(epoch_time)
        history['loss'].append(running_loss / len(loader))
        print(f'Result: DAE Epoch {epoch} finished in {epoch_time:.2f}s, Avg Loss: {running_loss/len(loader):.4f}')
        
    history['total_time'] = time.time() - start_total_time
    return history

def eval_classifier(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            # Map labels
            target = torch.where(target == 3, torch.tensor(0, device=device), target)
            target = torch.where(target == 5, torch.tensor(1, device=device), target)
            
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    acc = 100 * correct / total
    print(f'Accuracy: {acc:.2f}%')
    return acc
