import torch
import torchvision.models as models
import preprocessing
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 21)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(use_gpu=False, epochs=3, model_number=0, create_validation_dataloader=True):
    """Trains the specified model
    Args:
        use_gpu: whether the GPU should be used to train the model. Requires CUDA
        epochs: the number of epochs used to train the model
        model_number: enum of the model architecture. 0 is efficientnet_b0, 1 is efficientnet_b1, 2 is simple CNN
        create_validation_dataloader: toggles whether this outputs a part of the training set as a validation dataloader. This part doesn't get trained on
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
    else:
        model = Net()
        image_size = 128


    if create_validation_dataloader:
        train_loader, validation_loader = preprocessing.get_dataloaders(shuffled=True, image_side_length=image_size, augment_factor=4)
    else:
        train_loader = preprocessing.get_one_dataloader(shuffled=True, image_side_length=image_size, augment_factor=4)

    for parameter in model.parameters():
        parameter.requries_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.01,
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
        epoch_loss = 0.0

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
        
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)

        # Stop if relative improvement is too small
        if len(losses) > 1:
            relative_change = abs(losses[-2] - losses[-1]) / losses[-2]

            if relative_change < 1e-3:
                print(f"Stopping early at epoch {epoch}")
                epoch
                break

    torch.save(model.state_dict(), str(model_number) + "_" + str(epoch) + "_weights")
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