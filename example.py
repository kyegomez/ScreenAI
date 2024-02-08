import torch
from screenai.main import ScreenAI

# Create a tensor for the image
image = torch.rand(1, 3, 224, 224)

# Create a tensor for the text
text = torch.randint(0, 20000, (1, 1028))

# Create an instance of the ScreenAI model with specified parameters
model = ScreenAI(
    num_tokens = 20000,
    max_seq_len = 1028,
    patch_size=16,
    image_size=224,
    dim=512,
    depth=6,
    heads=8,
    vit_depth=4,
    multi_modal_encoder_depth=4,
    llm_decoder_depth=4,
    mm_encoder_ff_mult=4,
)

# Perform forward pass of the model with the given text and image tensors
out = model(text, image)

# Print the shape of the output tensor
print(out)
