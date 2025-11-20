import torch
import torch.nn as nn
from torchtyping import TensorType

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.
class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.gen_k = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.gen_q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.gen_v = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # B*T*E -> B*T*A
        k = self.gen_k(embedded)
        q = self.gen_q(embedded)
        v = self.gen_v(embedded)
        _, token_dim, attention_dim = q.shape

        attention_mat = (q @ torch.transpose(k, 1, 2)) / (attention_dim ** 0.5)
        tri_mask = torch.tril(torch.ones(token_dim, token_dim))
        bool_mask = tri_mask == 0
        attention_mat = attention_mat.masked_fill(bool_mask, float('-inf'))
        attention_mat = torch.nn.functional.softmax(attention_mat, dim=-1)

        # Return your answer to 4 decimal places
        return torch.round(attention_mat @ v, decimals=4)
