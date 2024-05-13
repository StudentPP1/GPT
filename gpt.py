import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, head_count):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.head_count = head_count

        self.query_layers = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(head_count)])
        self.key_layers = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(head_count)])
        self.value_layers = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(head_count)])
        self.fc_out = nn.Linear(head_count * embed_size, embed_size)

    def forward(self, embeddings):
        batch_size, token_count = embeddings.shape[:2]
        qkvs = torch.zeros(self.head_count, 3, batch_size, token_count, self.embed_size).to(embeddings.device)

        for i in range(self.head_count):
            qkvs[i, 0] = self.query_layers[i](embeddings)
            qkvs[i, 1] = self.key_layers[i](embeddings)
            qkvs[i, 2] = self.value_layers[i](embeddings)

        energy = torch.zeros(self.head_count, batch_size, token_count, token_count).to(embeddings.device)
        mask = torch.triu(torch.ones((token_count, token_count)), diagonal=1).bool()

        for h in range(self.head_count):
            for b in range(batch_size):
                for i in range(token_count):
                    for j in range(token_count):
                        energy[h, b, i, j] = torch.dot(qkvs[h, 0, b, i], qkvs[h, 1, b, j])
                energy[h, b] = energy[h, b].masked_fill(mask, float('-inf'))

        attention = torch.nn.functional.softmax(energy, dim=3)

        out = torch.zeros(batch_size, token_count, self.head_count, self.embed_size).to(embeddings.device)
        for h in range(self.head_count):
            for b in range(batch_size):
                for i in range(token_count):
                    for j in range(token_count):
                        out[b, i, h] += (attention[h, b, i, j] * qkvs[h, 2, b, j])

        out = out.reshape(batch_size, token_count, self.head_count * self.embed_size)
        return self.fc_out(out)


class Block(nn.Module):
    def __init__(self, embed_size, head_count):
        super(Block, self).__init__()
        self.attention = SelfAttention(embed_size, head_count)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, embeddings):
        attention = self.attention(embeddings)
        out = self.norm1(attention + embeddings)
        out = attention + self.feed_forward(out)
        out = self.norm2(out)
        return out


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, head_count):
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)

        self.layers = nn.ModuleList(
            [Block(embed_size, head_count) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, input_tokens, mask=None):
        batch_size, token_count = input_tokens.shape[:2]
        out = self.word_embedding(input_tokens)

        positions = torch.arange(0, token_count).expand(batch_size, token_count).to(input_tokens.device)
        position_encoding = self.position_encoding(positions, self.embed_size)
        out += position_encoding.reshape(out.shape)

        for layer in self.layers:
            out = layer(out)

        out = self.fc_out(out[:, -1, :].reshape(batch_size, self.embed_size)).reshape(batch_size, self.vocab_size)
        return torch.nn.functional.softmax(out, dim=1)

    def position_encoding(self, positions, embed_size):
        angle_rads = self.get_angles(
            positions.unsqueeze(2).float(),
            torch.arange(embed_size)[None, None, :].float().to(positions.device),
            embed_size
        )
        sines = torch.sin(angle_rads[:, :, 0::2])
        cosines = torch.cos(angle_rads[:, :, 1::2])
        pos_encoding = torch.cat([sines, cosines], dim=-1)
        pos_encoding = pos_encoding[None, ...]
        return pos_encoding

    def get_angles(self, pos, i, embed_size):
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / embed_size)
        return pos * angle_rates