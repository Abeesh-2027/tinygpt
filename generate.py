import torch
import torch.nn.functional as F

from model import TinyGPT
from tokenizer import Tokenizer
from utils import *
from config import *

# Device
device = get_device()

# Load dataset (needed to rebuild tokenizer)
with open("dataset.txt","r",encoding="utf-8") as f:
    text = f.read()

# Recreate tokenizer
tokenizer = Tokenizer(text)

# Create model
model = TinyGPT(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer
)

model = model.to(device)

# Load trained weights
model = load_model(
    model,
    "model.pt",
    device
)

model.eval()

# Generate function
def generate(model,prompt,max_new_tokens=200):

    tokens = tokenizer.encode(prompt)

    x = torch.tensor(tokens,dtype=torch.long)

    x = x.unsqueeze(0).to(device)

    for _ in range(max_new_tokens):

        x_cond = x[:,-block_size:]

        logits,loss = model(x_cond)

        logits = logits[:,-1,:]

        probs = F.softmax(logits,dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        x = torch.cat(
            (x,next_token),
            dim=1
        )

    output = tokenizer.decode(
        x[0].tolist()
    )

    return output


# Interactive loop
print("TinyGPT ready!")
print("Type 'exit' to quit")

while True:

    prompt = input("\nEnter prompt: ")

    if prompt == "exit":
        break

    result = generate(
        model,
        prompt,
        200
    )

    print("\nOutput:\n")
    print(result)
