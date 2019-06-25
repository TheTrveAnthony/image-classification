import os
import torch.onnx
import torchvision.models

##### This script will download resnet 18 and convert it into onnx, comment the last line to keep the pth file
##### For other models go here : https://pytorch.org/docs/stable/torchvision/models.html 

model = torchvision.models.resnet18(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)
nom = "resnet18.onnx"
torch.onnx.export(model, dummy_input, nom)
os.system("wget https://raw.githubusercontent.com/onnx/models/master/models/image_classification/synset.txt")
os.system("rm ~/.cache/torch/checkpoints/resnet18-5c106cde.pth")