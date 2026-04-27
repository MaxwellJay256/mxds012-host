import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM 模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 平均池化
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算通道注意力
        ca = self.channel_attention(x) * x

        # 计算空间注意力
        max_pool = torch.max(ca, dim=1, keepdim=True)[0]  # 最大池化
        avg_pool = torch.mean(ca, dim=1, keepdim=True)  # 平均池化
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa_input) * ca

        return sa

# CBAM-ConvLSTM 单元
class CBAMConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(CBAMConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = (1, kernel_size)
        self.padding = (0, kernel_size // 2)
        self.bias = bias

        # CBAM 注意力机制
        self.cbam = CBAM(hidden_dim)

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 调整 h_cur 和 c_cur 以匹配 input_tensor 的尺寸
        if h_cur.shape[2:] != input_tensor.shape[2:]:
            h_cur = F.interpolate(h_cur, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
            c_cur = F.interpolate(c_cur, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        # 在 h_next 上应用 CBAM 注意力
        h_next = self.cbam(h_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# CBAM-LSTM 网络
class CBAMLSTMNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CBAMLSTMNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))

        # CBAM-ConvLSTM 单元
        self.convLSTM = CBAMConvLSTMCell(input_dim=32, hidden_dim=32, kernel_size=23)
        self.final_pool = nn.AdaptiveAvgPool2d((4, 1))  # 固定池化输出尺寸，避免 ONNX 导出时 output_size 非常量

        # 由 LazyLinear 自动推断 in_features
        self.fc1 = nn.LazyLinear(64)

        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        b, t_seq, c, h, w = x.size()
        h_t, c_t = self.convLSTM.init_hidden(batch_size=b, image_size=(h, w))

        for t in range(t_seq):
            x_t = self.conv1(x[:, t, :, :, :])
            x_t = F.relu(x_t)
            x_t = self.pool1(x_t)
            h_t, c_t = self.convLSTM(x_t, (h_t, c_t))

        # 高度固定为 4，宽度压缩为 1。
        last_output = self.final_pool(h_t)
        last_output = torch.flatten(last_output, start_dim=1)
        x = F.relu(self.fc1(last_output))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
