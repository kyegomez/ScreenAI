[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Screen AI
Implementation of the ScreenAI model from the paper: "A Vision-Language Model for UI and Infographics Understanding"

## Install
`pip3 install screenai`

## Usage
```python

import torch
from screenai.main import ScreenAI

# Create a tensor
image = torch.rand(1, 3, 224, 224)
text = torch.randn(1, 1, 512)

# Model
model = ScreenAI(
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


# Forward
out = model(text, image)

# Print the output shape
print(out.shape)


```

# License
MIT
