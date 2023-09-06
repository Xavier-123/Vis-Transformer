import torch
import torch.nn as nn
import torch.nn.functional as F

from FFN import SublayerConnection

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, mutual_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.mutual_attn = mutual_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList(SublayerConnection(d_model, dropout) for _ in range(3))

    def forward(self, memory, x, source_mask, target_mask):
        # 将x传入第一个子层结构，第一个子层结构的输入分别是x和self-attn函数，因为是自注意力机制，所以Q,K,V都是x，
        # 最后一个参数是目标数据掩码张量，这时要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据，
        # 比如在解码器准备生成第一个字符或词汇时，我们其实已经传入了第一个字符以便计算损失，
        # 但是我们不希望在生成第一个字符时模型能利用这个信息，因此我们会将其遮掩，同样生成第二个字符或词汇时，
        # 模型只能使用第一个字符或词汇信息，第二个字符以及之后的信息都不允许被模型使用.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入x; k，v是编码层输出memory，
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄漏，而是遮蔽掉对结果没有意义的字符而产生的注意力值，
        # 以此提升模型效果和训练速度. 这样就完成了第二个子层的处理.
        x = self.sublayer[1](x, lambda x: self.mutual_attn(x, memory, memory, target_mask))

        # 最后一个子层就是前馈全连接子层，经过它的处理后就可以返回结果.这就是我们的解码器层结构.
        x = self.sublayer[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList(layer for _ in range(N))
        self.norm = nn.LayerNorm(layer.d_model)
    def forward(self, memory, x, source_mask, target_mask):
        for layer in self.layers:
            x = layer(memory, x, source_mask, target_mask)
        return self.norm(x)


'''输出head'''
class Generator(nn.Module):
    def __init__(self, d_model, out):
        super().__init__()
        self.project = nn.Linear(d_model, out)

    def forward(self, x):
        x = F.log_softmax(self.project(x), dim=-1)
        return x


if __name__ == '__main__':
    pass