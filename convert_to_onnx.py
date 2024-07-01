import torch
from models import *

model_path = 'out/model.pth'

model = CNN()
model.load_state_dict(torch.load(model_path))

dummy_input = torch.rand(280*280*4)


torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=["input"],
    dynamic_axes={'input': {0: 'batch_size'}}
)