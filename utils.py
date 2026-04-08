import torch
import random

# Detect device
def get_device():

    if torch.cuda.is_available():
        return 'cuda'

    return 'cpu'


# Create train/validation split
def train_val_split(data,split=0.9):

    n=int(split*len(data))

    train_data=data[:n]

    val_data=data[n:]

    return train_data,val_data


# Batch generator
def get_batch(data,block_size,batch_size,device):

    ix=torch.randint(
        len(data)-block_size,
        (batch_size,)
    )

    x=torch.stack([
        data[i:i+block_size]
        for i in ix
    ])

    y=torch.stack([
        data[i+1:i+block_size+1]
        for i in ix
    ])

    x=x.to(device)

    y=y.to(device)

    return x,y


# Estimate loss without training
@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    eval_iters,
    block_size,
    batch_size,
    device
):

    model.eval()

    losses={}

    for split in ['train','val']:

        data=train_data if split=='train' else val_data

        loss_list=torch.zeros(eval_iters)

        for k in range(eval_iters):

            X,Y=get_batch(
                data,
                block_size,
                batch_size,
                device
            )

            logits,loss=model(X,Y)

            loss_list[k]=loss.item()

        losses[split]=loss_list.mean()

    model.train()

    return losses


# Save model
def save_model(model,path="model.pt"):

    torch.save(
        model.state_dict(),
        path
    )


# Load model
def load_model(model,path="model.pt",device='cpu'):

    model.load_state_dict(
        torch.load(
            path,
            map_location=device
        )
    )

    return model


# Seed for reproducibility
def set_seed(seed=42):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
