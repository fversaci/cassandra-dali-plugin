import torch

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

model = (
    torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
    .eval()
)
traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
torch.jit.save(traced_model, "model.pt")
