import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse

from .data import get_cat_dog_dataloaders, get_pretrain_dataloaders
from .models import SimpleCNN, DAE, FineTunedClassifier
from .train import train_classifier, train_dae, eval_classifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_baseline(epochs=10, fractions=[1.0]):
    print(f"\n--- Running Baseline CNN (Epochs: {epochs}, Fractions: {fractions}) ---")
    
    results = {}
    
    for frac in fractions:
        print(f"\nTraining Baseline with {frac*100}% of data...")
        train_loader, test_loader = get_cat_dog_dataloaders(batch_size=64, fraction=frac)
        
        model = SimpleCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        history = train_classifier(model, train_loader, criterion, optimizer, DEVICE, epochs=epochs)
        final_acc = eval_classifier(model, test_loader, DEVICE)
        
        history['final_acc'] = final_acc
        results[frac] = history
    
    # Save baseline results
    with open(os.path.join(RESULTS_DIR, 'baseline_results.json'), 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f)
        
    return results

def run_pretraining(epochs=20):
    print(f"\n--- Running DAE Pre-training (Epochs: {epochs}) ---")
    # Train on non-cat/dog classes
    train_loader, _ = get_pretrain_dataloaders(batch_size=64)
    
    model = DAE(latent_dim=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    history = train_dae(model, train_loader, criterion, optimizer, DEVICE, epochs=epochs, noise_std=0.1)
    
    # Save the pre-trained encoder
    torch.save(model.encoder.state_dict(), os.path.join(RESULTS_DIR, "pretrained_encoder.pth"))
    
    with open(os.path.join(RESULTS_DIR, 'dae_history.json'), 'w') as f:
        json.dump(history, f)
        
    return model, history

def run_finetuning(pretrained_encoder_path, fractions=[0.1, 0.25, 0.5, 1.0], epochs=10, lr=0.001):
    print(f"\n--- Running Fine-tuning Experiments (Epochs: {epochs}, Fractions: {fractions}, LR: {lr}) ---")
    
    if not os.path.exists(pretrained_encoder_path):
        print(f"Error: Pretrained encoder not found at {pretrained_encoder_path}. Run --mode pretrain first.")
        return {}
        
    results = {}
    
    for frac in fractions:
        print(f"\nTraining with {frac*100}% of data...")
        train_loader, test_loader = get_cat_dog_dataloaders(batch_size=64, fraction=frac)
        
        # Load Pre-trained Encoder
        encoder = DAE(latent_dim=256).encoder # Instantiate fresh encoder structure
        encoder.load_state_dict(torch.load(pretrained_encoder_path))
        
        # Create Fine-tuned Model
        model = FineTunedClassifier(encoder).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = train_classifier(model, train_loader, criterion, optimizer, DEVICE, epochs=epochs)
        final_acc = eval_classifier(model, test_loader, DEVICE)
        
        history['final_acc'] = final_acc
        results[frac] = history
        
    with open(os.path.join(RESULTS_DIR, 'finetune_results.json'), 'w') as f:
        # Convert keys to strings for JSON
        json.dump({str(k): v for k, v in results.items()}, f)
        
    return results

def plot_results():
    print("\n--- Generating Plots ---")
    # Load results
    try:
        baseline_results = {}
        if os.path.exists(os.path.join(RESULTS_DIR, 'baseline_results.json')):
            with open(os.path.join(RESULTS_DIR, 'baseline_results.json'), 'r') as f:
                baseline_results = json.load(f)
        elif os.path.exists(os.path.join(RESULTS_DIR, 'baseline_history.json')):
             with open(os.path.join(RESULTS_DIR, 'baseline_history.json'), 'r') as f:
                baseline_results = {'1.0': json.load(f)}
        
        with open(os.path.join(RESULTS_DIR, 'finetune_results.json'), 'r') as f:
            finetune_results = json.load(f) # Keys are strings now
            
    except FileNotFoundError:
        print("Could not find result files. Run baseline and finetune experiments first.")
        return

    plt.figure(figsize=(12, 6))
    
    # Plot 1: Learning Curve (Show 100% for both if available)
    plt.subplot(1, 2, 1)
    if '1.0' in baseline_results:
        plt.plot(baseline_results['1.0']['loss'], label='Baseline (100% data)', marker='o')
    elif 1.0 in baseline_results:
        plt.plot(baseline_results[1.0]['loss'], label='Baseline (100% data)', marker='o')

    if '1.0' in finetune_results:
        plt.plot(finetune_results['1.0']['loss'], label='Fine-tuned (100% data)', marker='x')
    elif 1.0 in finetune_results: # Handle if somehow keys are floats
        plt.plot(finetune_results[1.0]['loss'], label='Fine-tuned (100% data)', marker='x')
        
    plt.title('Training Loss Convergence (100% Data)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Data Efficiency
    plt.subplot(1, 2, 2)
    
    # Helper to extract points
    def get_points(results_dict):
        fractions = []
        accs = []
        for k, v in results_dict.items():
            fractions.append(float(k))
            accs.append(v['final_acc'])
        sorted_pairs = sorted(zip(fractions, accs))
        return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]
    
    ft_fracs, ft_accs = get_points(finetune_results)
    plt.plot(ft_fracs, ft_accs, marker='o', label='Fine-tuned DAE')
    
    bl_fracs, bl_accs = get_points(baseline_results)
    plt.plot(bl_fracs, bl_accs, marker='s', linestyle='--', label='Baseline CNN')
    
    plt.title('Data Efficiency: Accuracy vs Data Fraction')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'comparison_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 DAE Experiments")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'baseline', 'pretrain', 'finetune', 'plot'], help='Experiment mode to run')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='Number of epochs for pre-training')
    parser.add_argument('--finetune_lr', type=float, default=0.001, help='Learning rate for fine-tuning')
    parser.add_argument('--data_fractions', type=float, nargs='+', default=[0.1, 0.25, 0.5, 1.0], help='Fractions of data for fine-tuning')
    
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    print(f"Results directory: {RESULTS_DIR}")
    
    if args.mode in ['all', 'baseline']:
        run_baseline(epochs=args.epochs, fractions=args.data_fractions)
        
    if args.mode in ['all', 'pretrain']:
        run_pretraining(epochs=args.pretrain_epochs)
        
    if args.mode in ['all', 'finetune']:
        pretrained_path = os.path.join(RESULTS_DIR, "pretrained_encoder.pth")
        run_finetuning(pretrained_path, fractions=args.data_fractions, epochs=args.epochs, lr=args.finetune_lr)
        
    if args.mode in ['all', 'plot']:
        plot_results()
