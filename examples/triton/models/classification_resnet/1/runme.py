import torch
import torchvision.models as models

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()

traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
torch.jit.save(traced_model, "model.pt")
