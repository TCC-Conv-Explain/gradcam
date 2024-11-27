from util import *
from GradCam import *
from glob import glob

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
inp_trans = transforms.Compose([transforms.Resize(512), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
view_trans=transforms.Compose([transforms.Resize(512), transforms.ToTensor()])


files=glob('./imagens/*')
images=[Image.open(f) for f in files]
tensors = [inp_trans(im).unsqueeze(0).to(device) for im in images]
view_tensors = [view_trans(im).unsqueeze(0).to(device) for im in images]
print(len(tensors))

gr = GradCAM(model)
heatmaps=[gr.forward(tensor) for tensor in tensors]

gb = GuidedBackPropagation(model)
bp = [gb.forward(tensor.requires_grad_()) for tensor in tensors]

for (idx, (tensor, output)) in enumerate(zip(view_tensors, heatmaps)):
  heatmap = output[0]
  im = (inp_to_np(tensor)//3)*2 + heatmap_to_np(heatmap) //3
  plt.title(output[1])
  plt.imshow(im)
  plt.savefig(f'./output/{files[idx][files[idx].rindex('/'):files[idx].rindex('.')]}-GC.png')

for (idx, (tensor, output)) in enumerate(zip(view_tensors, bp)):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
  fig.suptitle(output[1])
  ax1.imshow(inp_to_np(tensor))
  ax2.imshow(output[0], cmap='gray')
  plt.savefig(f'output/{files[idx][files[idx].rindex('/'):files[idx].rindex('.')]}-GB.png')

for (idx, (tensor, output_gc, output_bp)) in enumerate(zip(view_tensors, heatmaps, bp)):
  heatmap = output_gc[0]
  backpropagation = output_bp[0]
  heatmap = heatmap * backpropagation
  print(heatmap.max())
  heatmap = heatmap / heatmap.max()
  im = (inp_to_np(tensor)//3)*2 + heatmap_to_np(heatmap) //3
  plt.title(output_gc[1])
  plt.imshow(im)
  plt.savefig(f'output/{files[idx][files[idx].rindex('/'):files[idx].rindex('.')]}-GGC.png')