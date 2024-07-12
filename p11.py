import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute Q, K, V matrices
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)    # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)
        
        # Split the embedding into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Compute the weighted sum of the values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate the multiple heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Apply the final linear layer
        output = self.out(attn_output)
        
        return output

embed_dim = 128
num_heads = 8
seq_len = 10
batch_size = 32

x = torch.randn(batch_size, seq_len, embed_dim)
self_attention = SelfAttention(embed_dim, num_heads)
output = self_attention(x)
print(output.shape)  # Output shape will be (batch_size, seq_len, embed_dim)
