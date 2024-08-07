import wandb
import random

WANDB_ENTITY="mb-smle"
WANDB_PROJECT="git-testing"
with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT) as run:
    offset = random.random() / 5
    for epoch in range(1,20):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        t_loss = 2 ** -epoch + random.random() / epoch + offset
        wandb.log({"acc": acc, "loss": loss, "t_loss":t_loss})