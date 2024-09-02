import torch
import torch.nn.functional as F
from einops import einsum, rearrange

# 定义查询、键、值张量的形状：(batch_size, 序列长度, 头数量, 每个头的维度)
query = torch.randn(1, 256, 8, 64)  # 查询张量
key = torch.randn(1, 256, 2, 64)    # 键张量
value = torch.randn(1, 256, 2, 64)  # 值张量

# 定义每组中的头数量，本例中有2个kv头，意味着我们将有2组，每组4个头
num_head_groups = query.shape[2] // key.shape[2]

# 计算缩放因子
scale = query.size(-1) ** 0.5

# 交换序列长度和头数量的位置以加速计算
query = rearrange(query, "b n h d -> b h n d")
key = rearrange(key, "b s h d -> b h s d")
value = rearrange(value, "b s h d -> b h s d")

# 将查询头数量按组分割，引入额外的 'g' 维度
query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)

# 计算注意力得分，并在组维度上求和以执行平均
scores = einsum(query, key, "b g h n d, b h s d -> b g h n s")
attention = F.softmax(scores / scale, dim=-1)

# 应用权重到值头
out = einsum(attention, value, "b g h n s, b h s d -> b g h n d")

# 重塑回原始维度
out = rearrange(out, "b g h n d -> b n (h g) d")
print(out)