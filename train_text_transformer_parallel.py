import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
sys.path.append('/usr/xtmp/eyc14/project/')
from patent_dataset import *
from text_transformer_batch import *
import pandas as pd
from torch.nn.parallel import DataParallel

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
        encoder_params = (
            list(base_model.encoder_title.parameters()) +
            list(base_model.encoder_abstract.parameters()) +
            list(base_model.encoder_claims.parameters())
        )
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
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatentEmbeddingModel(device=device, model_name=model_name, freeze_encoder=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    # Training phases
    print("Starting training with frozen encoder...")
    train_model(model, device, train_loader, val_loader, epochs=3, fine_tune=False, 
               save_path="/usr/xtmp/eyc14/project/frozen_encoder_best.pt")
    
    print("\nStarting fine-tuning with unfrozen encoder...")
    train_model(model, device, train_loader, val_loader, epochs=3, fine_tune=True, 
               save_path="/usr/xtmp/eyc14/project/fine_tuned_best.pt")

if __name__ == "__main__":
    main()