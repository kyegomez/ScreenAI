import torch 
from screenai.main import ScreenAI

# Create a tensor
image = torch.rand(1, 3, 224, 224)
text = torch.randn(1, 1, 512)

# Model
model = ScreenAI(
    patch_size=(4, 6),
    image_size=224,
    dim=512,
    depth=6,
    heads=8,
    vit_depth=4,
    multi_modal_encoder_depth=4,
    llm_decoder_depth=4,
    mm_encoder_ff_mult=4
)


# Forward
out = model(image, text)

# Print the output shape
print(out.shape)
