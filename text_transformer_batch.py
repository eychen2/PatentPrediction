import torch
import torch.nn as nn
from transformers import AutoModel

def set_encoder_trainable(model, trainable):
    """Sets encoders trainable for a model (works with base_model or regular model)"""
    for encoder in [model.encoder_title, model.encoder_abstract, model.encoder_claims]:
        for param in encoder.parameters():
            param.requires_grad = trainable
            
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, token_embeddings, attention_mask):
        # attention_mask: [batch_size, seq_len]
        scores = self.attention(token_embeddings).squeeze(-1)  # [batch_size, seq_len]
        scores = scores.masked_fill(attention_mask == 0, -1e9)  # mask padded tokens
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)    # [batch_size, seq_len, 1]
        return torch.sum(token_embeddings * weights, dim=1)     # weighted sum over tokens

class PatentEmbeddingModel(nn.Module):
    def __init__(self, device, hidden=128, model_name="bert-base-uncased", num_outputs=1, freeze_encoder=True):
        super().__init__()
        self.device = device
        self.hidden = hidden
        
        # Separate encoders
        self.encoder_title = AutoModel.from_pretrained(model_name)
        self.encoder_abstract = AutoModel.from_pretrained(model_name)
        self.encoder_claims = AutoModel.from_pretrained(model_name)
        
        # Projection layers
        self.hidden_size = self.encoder_title.config.hidden_size
        self.proj_title = nn.Linear(self.hidden_size, hidden)
        self.proj_abstract = nn.Linear(self.hidden_size, hidden)
        self.proj_claims = nn.Linear(self.hidden_size, hidden)
        
        # Final regression
        self.regressor = nn.Linear(hidden * 3, num_outputs)
        self.pooling = AttentionPooling(self.hidden_size)

    def process_text_field(self, texts, encoder, proj_layer):
        """Process a batch of variable-length text chunks"""
        # Flatten all chunks across batch
        all_inputs = []
        for sample in texts:
            all_inputs.extend(sample["input_ids"])
        input_ids = torch.stack(all_inputs).to(self.device)
        
        all_masks = []
        for sample in texts:
            all_masks.extend(sample["attention_mask"])
        attention_mask = torch.stack(all_masks).to(self.device)
        
        # Get embeddings for all chunks
        with torch.set_grad_enabled(self.training):
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            chunk_embeddings = self.pooling(outputs.last_hidden_state, attention_mask)
        
        # Reconstruct original batch structure
        embeddings = []
        current_idx = 0
        for sample in texts:
            num_chunks = len(sample["input_ids"])
            sample_emb = chunk_embeddings[current_idx:current_idx+num_chunks].mean(dim=0)
            embeddings.append(sample_emb)
            current_idx += num_chunks
        
        return proj_layer(torch.stack(embeddings))

    def forward(self, title, abstract, claims):
        title_emb = self.process_text_field(title, self.encoder_title, self.proj_title)
        abstract_emb = self.process_text_field(abstract, self.encoder_abstract, self.proj_abstract)
        claims_emb = self.process_text_field(claims, self.encoder_claims, self.proj_claims)
        
        combined = torch.cat([title_emb, abstract_emb, claims_emb], dim=1)
        return self.regressor(combined)