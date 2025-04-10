
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from transformers import AutoTokenizer
import os
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
import torchvision.transforms.functional as F

class ResizePadToSquare:
    def __init__(self, target_size=224, fill=255):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, image: Image.Image):
        # Convert to grayscale if not L or RGB
        if image.mode not in ['L', 'RGB']:
            image = image.convert('L')

        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = F.resize(image, (new_h, new_w))

        # Padding to square
        if image.mode == 'L':
            new_image = Image.new("L", (self.target_size, self.target_size), self.fill)
        else:
            new_image = Image.new("RGB", (self.target_size, self.target_size), (self.fill,) * 3)

        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))

        # Convert grayscale to RGB if needed
        if new_image.mode == 'L':
            new_image = new_image.convert('RGB')

        return new_image


transform = T.Compose([
    ResizePadToSquare(target_size=224),  # your custom transform
    T.ToTensor(),                        # [0, 255] to [0.0, 1.0]
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to [-1, 1]
])

class PatentDataset(Dataset):
    def __init__(self, df, tokenizer, image_dir, transform, max_words=150, batch_size=16, max_length=512, label_column="cit_5yr_window"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.batch_size = batch_size
        self.max_length = max_length
        self.label_column = label_column

        # Group all images by patent ID
        self.transform = transform
        self.patent_to_images = {}
        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                patent_id = filename.split('_')[0]
                path = os.path.join(image_dir, filename)
                self.patent_to_images.setdefault(patent_id, []).append(path)
        
        # Patent ID list is the dataset index
        self.patent_ids = list(self.patent_to_images.keys())
        self.valid_indices = self._find_valid_indices()

    def chunk_text(self, text):
        words = text.split()
        return [' '.join(words[i:i + self.max_words]) for i in range(0, len(words), self.max_words)]

    def tokenize_chunks(self, text):
        chunks = self.chunk_text(text)
        # Tokenize all chunks at once
        inputs = self.tokenizer(chunks, padding='max_length', truncation=True, 
                              return_tensors='pt', max_length=self.max_length)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        
    def _find_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            patent_id = row["publication_number"]
            if (patent_id in self.patent_to_images and 
                len(self.patent_to_images[patent_id]) > 0):
                valid_indices.append(idx)
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        patent_id = row["publication_number"]
        image_paths = self.patent_to_images[patent_id]

        images = []
        for path in sorted(image_paths):  # sort for consistency
            try:
                img = Image.open(path).convert("RGB")  # Always force RGB
            except UnidentifiedImageError:
                continue
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Return stacked images [num_images, 3, H, W] and the patent ID
        image_tensor = torch.stack(images)
        return {
            "title": self.tokenize_chunks(row["title_localized"]),
            "abstract": self.tokenize_chunks(row["abstract_localized"]),
            "claims": self.tokenize_chunks(row["claims_localized"]),
            "description": self.tokenize_chunks(row["description_localized"]),
            "images": image_tensor,
            "label": torch.tensor(row[self.label_column], dtype=torch.float)
        }
