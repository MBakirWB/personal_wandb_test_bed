import wandb
import random

WANDB_ENTITY="mb-smle"
WANDB_PROJECT="mb-testing"
with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT) as run:
    # Simulate logging model metrics
    offset = random.random() / 5
    for epoch in range(1,20):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        wandb.log({"acc": acc, "loss": loss})