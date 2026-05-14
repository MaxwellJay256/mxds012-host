import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import onnx
from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
from CNN2DStaticModel import CNN2DStaticModel

# model = CNN2DStaticModel()
dummy_input = torch.randn(1, 1, 4, 500).to(device)
# torch.onnx.export(model, dummy_input, 'model.onnx')
# writer = SummaryWriter()
model = CNN2DStaticModel().to(device)
y = model(dummy_input)
# graph = make_dot(y, params=dict(model.named_parameters()))
# graph.render('model_graph', format='png')  # 输出PNG
# writer.add_graph(model, torch.randn(1, 1, 4, 500))
# writer.close()
# summary(model, input_size=(1, 4, 500), device=str(device))

