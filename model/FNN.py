import torch
import torch.nn as nn
import torch.nn.functional as f


class FNN(nn.Module):
    def __init__(self, pre_length, embed_size, feature_size, seq_length, hidden_size):
        super().__init__()
        self.embed_size = embed_size
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.scale = 0.1

        # Learnable embedding matrix for input features.
        # 输入特征的可学习嵌入矩阵。
        self.embeddings = nn.Parameter(torch.randn(feature_size, embed_size))

        # Learnable weights and biases for two layers of complex linear transformations.
        # 用于两个复线性变换层的可学习权重和偏置。
        self.w1 = nn.Parameter(self.scale * torch.randn(2, embed_size, embed_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, embed_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, embed_size, embed_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, embed_size))

        # Fully connected layers for downstream processing after Fourier transformation.
        # 傅里叶变换后的下游全连接层处理。
        self.fc = nn.Sequential(
            nn.Linear(seq_length * embed_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),     # Apply Layer Normalization. 应用层归一化。
            nn.LeakyReLU(),                    # LeakyReLU activation. 使用LeakyReLU激活函数。
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def tokenEmb(self, x):
        """Applies embedding by matrix multiplication between input and embedding matrix.
        通过输入与嵌入矩阵的乘法获得嵌入表示。
        """
        return torch.matmul(x, self.embeddings)

    def fourierGC(self, x):
        """Applies complex-valued graph convolution in the frequency domain.
        在频域中应用复数图卷积。
        """
        # First complex linear transformation.
        # 第一个复线性变换。
        o1_real = torch.einsum('bli,io->blo', x.real, self.w1[0]) - \
                  torch.einsum('bli,io->blo', x.imag, self.w1[1]) + self.b1[0]
        o1_imag = torch.einsum('bli,io->blo', x.imag, self.w1[0]) + \
                  torch.einsum('bli,io->blo', x.real, self.w1[1]) + self.b1[1]

        # Second complex linear transformation with ReLU activation.
        # 第二个复线性变换，并使用ReLU激活。
        o2_real = f.relu(torch.einsum('bli,io->blo', o1_real, self.w2[0]) - \
                         torch.einsum('bli,io->blo', o1_imag, self.w2[1]) + self.b2[0])
        o2_imag = f.relu(torch.einsum('bli,io->blo', o1_imag, self.w2[0]) + \
                         torch.einsum('bli,io->blo', o1_real, self.w2[1]) + self.b2[1])

        # Stack real and imaginary parts to form a complex tensor.
        # 将实部与虚部堆叠形成复数张量。
        return torch.view_as_complex(torch.stack([o2_real, o2_imag], dim=-1))

    def forward(self, x):
        """Forward pass through the FGN model.
        执行FGN模型的前向传播。
        """
        B, seq_len, _ = x.size()  # B: batch size, seq_len: 序列长度

        x = self.tokenEmb(x)  # Apply token embedding. 应用token嵌入。
        x = torch.fft.rfft(x, dim=1, norm='ortho')  # Apply real FFT along the time dimension. 在时间维度上进行实数FFT。
        x = self.fourierGC(x)  # Apply Fourier graph convolution. 应用傅里叶图卷积。
        x = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")  # Inverse FFT to return to time domain. 逆FFT返回时间域。

        return self.fc(x.reshape(B, -1))  # Flatten and pass through fully connected layers. 扁平化后输入全连接层。
