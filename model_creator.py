import random
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

def train_model(use_gpu=False, epochs=1, model_number=4, create_validation_dataloader=True, augment=True, reaugment_every_epoch=False, batch_size = 8, isRandom = False, pretrained = False):
    """Trains the specified model
    Args:
        use_gpu: whether the GPU should be used to train the model. Requires CUDA
        epochs: the number of epochs used to train the model
        model_number: enum of the model architecture. 0 is efficientnet_b0, 1 is efficientnet_b1, 2 is simple CNN
        create_validation_dataloader: toggles whether this outputs a part of the training set as a validation dataloader. This part doesn't get trained on
        augment: toggles whether the imageset is augmented
        reagument_every_epoch: toggles whether the data is augmented from scratch every epoch to reduce overfitting
        batch_size: batch size of the training dataloader
        isRandom: if False, sets a seed for torch and random
        pretrained: if True, uses default weights for EfficientNet
    Returns:
            the model, a validation set that wasn't trained on"""

    if not isRandom:
        torch.manual_seed(0)
        random.seed(0)

    weights = None
    if model_number == 0:
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            21 # Number of classes. Hardcoded as we don't need to change it ~Erik
        )
        image_size = 224
    elif model_number == 1:
        if pretrained:
            weights = models.EfficientNet_B1_Weights.DEFAULT
        model = models.efficientnet_b1(weights=weights)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            21
        )
        image_size = 224
    elif model_number == 2:
        if pretrained:
            weights = models.EfficientNet_B2_Weights.DEFAULT
        model = models.efficientnet_b2(weights=weights)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            21 
        )
        image_size = 224
    elif model_number == 3:
        if pretrained:
            weights = models.EfficientNet_B3_Weights.DEFAULT
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            21
        )
        image_size = 224
    else:
        model = Net()
        image_size = 128

    # At first, do a rough training of the new classifiers
    if pretrained:
        for param in model.features.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3
        )
    else:
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
        if torch.cuda.is_available() and (not isRandom):
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)  # for multi-GPU setups
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    model.train()

    if create_validation_dataloader:
        train_loader, validation_loader = preprocessing.get_dataloaders(shuffled=False, image_side_length=image_size, train_batch_size=batch_size, augment=augment)
    else:
        train_loader = preprocessing.get_one_dataloader(shuffled=False, image_side_length=image_size, batch_size=batch_size)


    losses = []
    validation_accuracy = []
    max_epoch_accuracy = 0
    for epoch in tqdm(range(epochs), desc='Epoch'):
        epoch_loss = 0.0

        # Then fine tune the model
        if epoch == 5 and pretrained:
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1.e-5
            )

        """
        if reaugment_every_epoch and (epoch == 0 or augment_factor >= 1) :
            if create_validation_dataloader:
                train_loader = torch.utils.data.DataLoader(preprocessing.Imageset(train=True,
                    storage=data_storage.augment(factor=augment_factor, copy=True, val_destructive=True)
                    ), batch_size=batch_size, shuffle=True)
            else:
                train_loader = torch.utils.data.DataLoader(preprocessing.ImagesetFull(
                    storage=data_storage.augment(factor=augment_factor, copy=True, val_destructive=True)
                    ), batch_size=batch_size, shuffle=True)
        """

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
        epoch_accuracy = evaluate_model(model, validation_loader)
        if epoch_accuracy > max_epoch_accuracy:
            torch.save(model.state_dict(), "Model number:" + str(model_number) + "_Epochs:" + str(epochs) + "_Pretrained:" + str(pretrained) + "_best_temp_weights")
            max_epoch_accuracy = epoch_accuracy
        
        validation_accuracy.append(evaluate_model(model, validation_loader))

        # Stop if relative improvement is too small
        if len(losses) > 1 and (not reaugment_every_epoch):
            relative_change = abs(validation_accuracy[-2] - validation_accuracy[-1]) / validation_accuracy[-2]

            if relative_change < 1e-3:
                epochs = epoch
                print("Stopping early at epoch " + str(epoch))
                break

    torch.save(model.state_dict(), "Model number:" + str(model_number) + "_Epochs:" + str(epochs) + "_Pretrained:" + str(pretrained) + "_weights")
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

def load_model():
    model = models.efficientnet_b1(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 21)
    model.load_state_dict(torch.load("1_20_0_weights", weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model