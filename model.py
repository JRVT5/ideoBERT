import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class PoliticalTextClassification(nn.Module):
    """Write your PoliticalTextClassifcation model here"""
    def __init__(self):
        super().__init__()

        self.hidden_dim = 100
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        bert_embedding_size = self.bert.config.hidden_size

        self.linear_layers = nn.Sequential(
            nn.Linear(bert_embedding_size, self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_states = bert_output.last_hidden_state

        cls_embeddings = last_hidden_states[:, 0, :]

        linear_output = self.linear_layers(cls_embeddings)

        output = F.sigmoid(linear_output)

        return output

