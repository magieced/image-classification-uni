import torch
import torchvision.models as models
import preprocessing
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # (256, 8, 8)

            nn.AdaptiveAvgPool2d((1, 1))    # (256, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 21)              # Hardcoded as we always have 21 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(use_gpu=False, epochs=1, model_number=4, create_validation_dataloader=True, augment_factor=0):
    """Trains the specified model
    Args:
        use_gpu: whether the GPU should be used to train the model. Requires CUDA
        epochs: the number of epochs used to train the model
        model_number: enum of the model architecture. 0 is efficientnet_b0, 1 is efficientnet_b1, 2 is simple CNN
        create_validation_dataloader: toggles whether this outputs a part of the training set as a validation dataloader. This part doesn't get trained on
        augment_factor: the increase in dataset size
    Returns:
            the model, a validation set that wasn't trained on"""
    if model_number == 0:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            21 # Number of classes. Hardcoded as we don't need to change it ~Erik
        )
        image_size = 224
    elif model_number == 1:
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            21 # Number of classes. Hardcoded as we don't need to change it
        )
        image_size = 224
    elif model_number == 2:
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            21 # Number of classes. Hardcoded as we don't need to change it
        )
        image_size = 224
    elif model_number == 3:
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            21 # Number of classes. Hardcoded as we don't need to change it
        )
        image_size = 224
    else:
        model = Net()
        image_size = 128

    if create_validation_dataloader:
        train_loader, validation_loader = preprocessing.get_dataloaders(shuffled=True, image_side_length=image_size, augment_factor=augment_factor)
    else:
        train_loader = preprocessing.get_one_dataloader(shuffled=True, image_side_length=image_size, augment_factor=augment_factor)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = 100
    )
    
    losses = []

    criterion = torch.nn.CrossEntropyLoss()

    if use_gpu:
        device = torch.device("cuda" if use_gpu else "cpu")
        model.to(device)
        criterion.cuda()

    model.train()

    losses = []
    for epoch in tqdm(range(epochs), desc='Epoch'):
        epoch_loss = 0.0

        for step, (example, label) in enumerate(tqdm(train_loader, desc='Batch')):
            if use_gpu:
                example = example.to(device)
                label = label.to(device)

            prediction = model(example)

            loss = criterion(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()

        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        print(epoch, epoch_loss / len(train_loader))

        # Stop if relative improvement is too small
        if len(losses) > 1:
            relative_change = abs(losses[-2] - losses[-1]) / losses[-2]

            if relative_change < 1e-3:
                epochs = epoch
                print(f"Stopping early at epoch {epoch}")
                break

    torch.save(model.state_dict(), str(model_number) + "_" + str(epochs) + "_" + str(augment_factor) + "_weights")
    if create_validation_dataloader:
        return model, validation_loader
    else:
        return model

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
    models.efficientnet_b0(weights = "[Model weights name]")
Replace the name of the model with whichever one you need (don't forget to import torchvision.models as models)
"""