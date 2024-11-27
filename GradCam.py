import torch
from util import *
class GradCAM:

  def __init__(self, model):
    self.activations=None
    self.gradients=None
    self.model=model
    self.model.eval()
    def forward_hook(module, input, output):
      self.activations = output

    def backward_hook(module, input, output):
      self.gradients = output[0]


    model._modules['layer4'].register_forward_hook(forward_hook)
    model._modules['layer4'].register_backward_hook(backward_hook)


  def forward(self, im):
    self.model.zero_grad()
    output = self.model(im)
    label = output.argmax().item()
    output[0,label].backward()
    alpha = self.gradients.squeeze(0).mean(dim=(1, 2))
    print(categories_MNIST[label])
    heatmap =  (self.activations.squeeze(0) * alpha.view(-1, 1, 1)).sum(dim=0)
    heatmap = transforms.Resize(im.shape[2:])(heatmap.unsqueeze(0))/heatmap.max()
    return (torch.clamp(heatmap, min=0).cpu(), categories_MNIST[label])



class GuidedBackPropagation:
  def __init__(self, model):
    self.activations=None
    self.gradients=None
    self.model=model
    self.model.eval()

    def backward_hook(module, input, output):
      if isinstance(module, torch.nn.ReLU):
        return (torch.clamp(input[0], min=0),)



    for i, module in enumerate(model.modules()):
      if isinstance(module, torch.nn.ReLU):
        module.inplace = False
        module.register_full_backward_hook(backward_hook)


  def forward(self, im):
    self.model.zero_grad()
    output = self.model(im)
    label = output.argmax().item()
    print(label)
    output[0,label].backward()
    ret = im.grad.clone()
    ret = ret.squeeze(0).sum(0)
    ret = (ret - ret.min())/(ret.max() - ret.min())
    return (ret.cpu(), categories_MNIST[label])

    