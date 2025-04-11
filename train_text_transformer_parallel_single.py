import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
sys.path.append('/usr/xtmp/eyc14/project/')
from patent_dataset import *
from text_transformer_batch_single import *
import pandas as pd
import numpy as np
from torch.nn.parallel import DataParallel
def evaluate_model(model, data_loader, device):
    """Compute MSE and R² for a given DataLoader."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            # Forward pass (matches training script's input format)
            outputs = model(
                title=batch["title"],
                abstract=batch["abstract"],
                claims=batch["claims"]
            ).squeeze().cpu().numpy()

            labels = batch["label"].cpu().numpy()
            predictions.append(outputs)
            targets.append(labels)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return mse, r2

def run_evaluation(model, device, train_loader, val_loader, test_loader, model_path):
    """Load saved model and evaluate on all splits."""
    # Load saved state (handles DataParallel)
    checkpoint = torch.load(model_path)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model with validation loss: {checkpoint['val_loss']:.4f}")

    # Evaluate on all datasets
    train_mse, train_r2 = evaluate_model(model, train_loader, device)
    val_mse, val_r2 = evaluate_model(model, val_loader, device)
    test_mse, test_r2 = evaluate_model(model, test_loader, device)

    print("\n=== Evaluation Results ===")
    print(f"Train  | MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"Val    | MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
    print(f"Test   | MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

    return {
        'train': {'mse': train_mse, 'r2': train_r2},
        'val': {'mse': val_mse, 'r2': val_r2},
        'test': {'mse': test_mse, 'r2': test_r2}
    }



def get_splits(dataset, train_size, val_size, test_size, seed=42):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Create a generator with the fixed seed
    generator = torch.Generator().manual_seed(seed)
    
    # Perform the split with the fixed generator
    return random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
def collate_fn(batch):
    # Custom collate function to handle our data structure
    return {
        "title": [item["title"] for item in batch],
        "abstract": [item["abstract"] for item in batch],
        "claims": [item["claims"] for item in batch],
        "description": [item["description"] for item in batch],
        "images": [item["images"] for item in batch],
        "label": torch.stack([item["label"] for item in batch])
        
    }


def train_model(model, device, train_loader, val_loader, epochs=3, lr=1e-4, fine_tune=False, save_path="best_model.pt"):
   # Adjust encoder trainability (works with or without DataParallel)
    base_model = model.module if isinstance(model, DataParallel) else model

    set_encoder_trainable(base_model, fine_tune)

    if fine_tune:
        optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr)
    else:
        # Handle both normal and DataParallel cases
        encoder_params = list(base_model.encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        non_encoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
        optimizer = torch.optim.AdamW(non_encoder_params, lr=lr)

    criterion = nn.MSELoss()
    model.train()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

        for batch in progress_bar:
            optimizer.zero_grad()
            outputs = model(
                title=batch["title"],
                abstract=batch["abstract"],
                claims=batch["claims"]
            ).squeeze()
            loss = criterion(outputs, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(train_loss=loss.item())

        train_loss = total_loss / len(train_loader)

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    title=batch["title"],
                    abstract=batch["abstract"],
                    claims=batch["claims"]
                ).squeeze()
                loss = criterion(outputs, batch["label"].to(device))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save model if validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'model_state_dict': state_dict,
                'val_loss': val_loss
                }, save_path)
            print(f"✅ New best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

def main():
    # Load data
    merged_df = pd.read_pickle('/usr/xtmp/eyc14/project/patent_text.pkl')
    
    # Setup transforms
    transform = T.Compose([
        ResizePadToSquare(target_size=224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Initialize tokenizer and dataset
    model_name = 'AI-Growth-Lab/PatentSBERTa'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = PatentDataset(merged_df, tokenizer, '/usr/xtmp/eyc14/project/data/patent_image/all_images', transform, batch_size=16)
    
    # Split dataset
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = get_splits(
    dataset, 
    train_size=train_size, 
    val_size=val_size, 
    test_size=test_size,
    seed=42  # Same seed ensures same splits every time
    )
    
    save_path="/usr/xtmp/eyc14/project/model/single/frozen_encoder_best_10.pt"
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=16, shuffle=False, collate_fn=collate_fn)
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatentEmbeddingModel(device=device, model_name=model_name, freeze_encoder=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    # Training phases
    print("Starting training with frozen encoder...")
    train_model(model, device, train_loader, val_loader, epochs=10, fine_tune=False, 
               save_path=save_path)
    
    # print("\nStarting fine-tuning with unfrozen encoder...")
    # train_model(model, device, train_loader, val_loader, epochs=3, fine_tune=True, 
    #            save_path="/usr/xtmp/eyc14/project/model/parallel/fine_tuned_best_gc_3.pt")
    checkpoint = torch.load(save_path)
    if all(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    eval_model = PatentEmbeddingModel(device=device, model_name=model_name, freeze_encoder=True)
    eval_model.load_state_dict(checkpoint, strict=False)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        eval_model = nn.DataParallel(eval_model)

    eval_model = eval_model.to(device)
    # Example usage
    print("running evaluation")
    results = run_evaluation(
        model=eval_model,
        device=device,
        train_loader=train_loader,  # Your DataLoaders
        val_loader=val_loader,
        test_loader=test_loader,
        model_path=save_path
    )

if __name__ == "__main__":
    main()