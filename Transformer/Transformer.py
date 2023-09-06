import torch
import torch.nn as nn

from Encoder import Encoder, EncoderLayer
from Decoder import Decoder, DecoderLayer, Generator
from Attention import MultiHeadedAttention, subsequent_mask
from FFN import PositionwiseFeedForward
from Position import PositionalEncoding, Embeddings
import copy


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        memory = self.encoder(source, source_mask)
        x = self.decoder(memory, target, source_mask, target_mask)
        return x

    def encode(self, source, source_mask):
        """编码函数, 以source和source_mask为参数"""
        # 使用src_embed对source做处理, 然后和source_mask一起传给self.encoder
        x = self.encoder(self.source_embed(source), source_mask)
        return x

    def decode(self, memory, source_mask, target, target_mask):
        """解码函数, 以memory即编码器的输出, source_mask, target, target_mask为参数"""
        # 使用tgt_embed对target做处理, 然后和source_mask, target_mask, memory一起传给self.decoder
        x = self.decoder(self.target_embed(memory), source_mask, target, target_mask)
        return x

def make_model(source_vocab, target_vocab, N=6,
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    # 实例化了多头注意力类，得到对象attn
    attn = MultiHeadedAttention(head, d_model)

    # 然后实例化前馈全连接类，得到对象ff
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码类，得到对象position
    position = PositionalEncoding(d_model, dropout)

    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层.
    # 在编码器层中有attention子层以及前馈全连接子层，
    # 在解码器层中有两个attention子层以及前馈全连接层.
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# class Transformer(nn.Module):
#     def __init__(self, d_model=512, d_ff=64, head=8, N=3, dropout=0.1):
#         super().__init__()
#         position_encoder = PositionalEncoding(d_model, dropout)
#         position_decoder = PositionalEncoding(d_model, dropout)
#         self_attn_encoder = MultiHeadedAttention(head, d_model)
#         self_attn_decoder = MultiHeadedAttention(head, d_model)
#         mutual_attn = MultiHeadedAttention(head, d_model)
#         feed_forward_encoder = PositionwiseFeedForward(d_model, d_ff)
#         feed_forward_decoder = PositionwiseFeedForward(d_model, d_ff)
#
#         self.ecoder_layer = EncoderLayer(d_model, self_attn_encoder, feed_forward_encoder)
#         self.encoder = Encoder(self.ecoder_layer, N)
#
#         self.decoder_layer = DecoderLayer(d_model, self_attn_decoder, mutual_attn, feed_forward_decoder)
#         self.decoder = Decoder(self.decoder_layer, N)
#
#
#         # Embeddings(d_model, source_vocab) 可以看成是一张图像的tensor
#         self.embeddings1 = nn.Sequential(Embeddings(d_model, source_vocab), position_encoder)
#         self.embeddings2 = nn.Sequential(Embeddings(d_model, source_vocab), position_decoder)
#
#         self.out = Generator(d_model, target_vocab)
#
#     def forward(self, x):
#         memory = self.encoder(x)
#         x = self.decoder(x, memory, source_mask, target_mask)
#         out = self.out(x)
#         return out

if __name__ == '__main__':
    # model = Transformer()
    # data = torch.randn([1, 4, 512])
    # print(model(data))

    model = make_model(11, 11, 1)
    print(model)

    source = torch.randn(1, 4, 512)
    target = torch.randn(1, 4, 512)

    source_mask = subsequent_mask(4)
    target_mask = subsequent_mask(4)
    model(source, target, source_mask, target_mask)