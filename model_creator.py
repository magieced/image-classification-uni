import torch
from torchvision.models import efficientnet_b0
import preprocessing
from tqdm import tqdm

def train_model(use_gpu = False, epochs = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        21 # Number of classes. Hardcoded as we don't need to change it ~Erik
    )

    train_loader, validation_loader = preprocessing.get_dataloaders()

    for parameter in model.parameters():
        parameter.requries_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.1,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = 100
    )
    
    losses = []

    criterion = torch.nn.CrossEntropyLoss()

    if use_gpu:
        model.cuda()
        criterion.cuda()

    losses = []
    for epoch in tqdm(range(epochs), desc='Epoch'):
        for step, (example, label) in enumerate(tqdm(train_loader, desc='Batch')):
            if use_gpu:
                example = example.cuda()
                label = label.cuda()

            example = example.float()

            optimizer.zero_grad()

            prediction = model(example)

            loss = criterion(prediction, label)

            losses.append(loss.item())

            loss.backward()

    torch.save(model.state_dict(), "efficientnet_weights.pth")

    return model, validation_loader

def evaluate_model(model, validation_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct/total
    print("Accuracy:", accuracy)
    return accuracy

"""
To load a model, do the following:
    efficientnet_b0(weights = "efficientnet_weights")
Replace the name of the model with whichever one you need (don't forget to load from torchvision.models)
"""