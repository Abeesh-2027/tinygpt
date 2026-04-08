import torch
from model import TinyGPT
from tokenizer import Tokenizer
from utils import *
from config import *

# Set random seed
set_seed()

# Device
device = get_device()
print("Using device:", device)

# Load dataset
with open("dataset.txt","r",encoding="utf-8") as f:
    text = f.read()

# Build tokenizer
tokenizer = Tokenizer(text)

# Encode dataset
data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

# Split data
train_data, val_data = train_val_split(data)

# Create model
model = TinyGPT(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer
)

model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print("Training started...")

# Training loop
for iter in range(max_iters):

    # Evaluate loss sometimes
    if iter % eval_interval == 0:

        losses = estimate_loss(
            model,
            train_data,
            val_data,
            50,
            block_size,
            batch_size,
            device
        )

        print(
            f"step {iter} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )

    # Get batch
    xb,yb = get_batch(
        train_data,
        block_size,
        batch_size,
        device
    )

    # Forward
    logits,loss = model(xb,yb)

    # Backprop
    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()


print("Training finished")

# Save model
save_model(model)

print("Model saved as model.pt")
