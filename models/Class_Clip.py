import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from models.LongCLIP.model import longclip


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def encode(self, *args, **kwargs):
        raise NotImplementedError





























    

class FrozenLongCLIPEmbedder(AbstractEncoder):

    def __init__(self, ckpt_path, device="cuda"):
        super().__init__()
        self.device = device

        
        model, _ = longclip.load(ckpt_path, device=device)
        self.model = model

        self.freeze()

    def freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, text):
        
        if isinstance(text, str):
            text = [text]

        
        tokens = longclip.tokenize(text).to(self.device)

        
        
        
        z = self.model.encode_text_full(tokens)  

        return z

    def encode(self, text):
        return self(text)
    
