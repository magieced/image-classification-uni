import random
import torch
import torchvision.models as models
import preprocessing
import numpy as np
from tqdm import tqdm
import torch.nn as nn

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

def train_model(use_gpu:bool=True, epochs:int=1, model_number:int=4, create_validation_dataloader:bool=True, augment:bool=True, batch_size:int=8, isRandom:bool=False, pretrained:bool=False):
    """Trains the specified model
    Also saves the model with the best accuracy on the validation dataset
    Args:
        use_gpu: toggles whether the GPU should be used to train the model. If CUDA is not available, defaults back to using the CPU
        epochs: the number of epochs used to train the model
        model_number: enum of the model architecture. 0 is efficientnet_b0, 1 is efficientnet_b1, 2 is simple CNN
        create_validation_dataloader: if True, splits off a part of the training set as a validation set. Otherwise, loads the provided validation set
        augment: toggles whether the imageset is augmented
        batch_size: batch size of the training dataloader
        isRandom: if False, sets a seed for torch and random
        pretrained: if True, uses default weights for EfficientNet and switches to fine tuning. Doesn't affect the simple CNN
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

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    use_gpu = use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)
    if use_gpu:
        criterion.cuda()
        if (not isRandom):
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)  # for multi-GPU setups
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    model.train()

    if create_validation_dataloader:
        train_loader, validation_loader = preprocessing.get_dataloaders(shuffled=isRandom, image_side_length=image_size, train_batch_size=batch_size, augment=augment)
    else:
        train_loader = preprocessing.get_one_dataloader(shuffled=isRandom, image_side_length=image_size, batch_size=batch_size)
        validation_loader = preprocessing.get_validation_dataloader(image_side_length=image_size, batch_size=batch_size)

    losses = []
    validation_accuracy = []
    max_epoch_accuracy = 0
    weight_path = "Model" + str(model_number) + "_Epochs" + str(epochs) + "_Pretraind" + str(pretrained) + "_best_temp_weights"

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
        #print(epoch, epoch_loss / len(train_loader))
        epoch_accuracy = evaluate_model(model, validation_loader)
        if epoch_accuracy > max_epoch_accuracy:
            torch.save(model.state_dict(), weight_path)
            max_epoch_accuracy = epoch_accuracy
        
        validation_accuracy.append(evaluate_model(model, validation_loader))

        # Stop if relative improvement is too small
        if len(losses) > 1 and (epoch > 5 or (not pretrained)):
            # We use -1 and -2 to catch any possible small improvements after the model has already started overfitting
            relative_change = abs(validation_accuracy[-1]) - abs(validation_accuracy[-2])

            if relative_change < 0:
                epochs = epoch
                print("Stopping early at epoch " + str(epoch))
                break

    model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
                          
    return model, validation_loader

def evaluate_model(model, validation_loader)->float:
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
    model.load_state_dict(torch.load("bestyolob1pretrain", weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model

def load_multiple_models():
    """Outputs a list of pretrained models as a list, namely B0 through B3, with two weights per model.
    Returns: A list of models set into eval
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = []

    model_list.append(models.efficientnet_b0(weights=None))
    model_list[0].classifier[1] = torch.nn.Linear(1280, 21)
    model_list[0].load_state_dict(torch.load("Model0_Epochs20_PretraindTrue_best_temp_weights", weights_only=True, map_location=device))

    model_list.append(models.efficientnet_b0(weights=None))
    model_list[1].classifier[1] = torch.nn.Linear(1280, 21)
    model_list[1].load_state_dict(torch.load("bestyolob0pretrain", weights_only=True, map_location=device))

    model_list.append(models.efficientnet_b1(weights=None))
    model_list[2].classifier[1] = torch.nn.Linear(1280, 21)
    model_list[2].load_state_dict(torch.load("Model1_Epochs20_PretraindTrue_best_temp_weights", weights_only=True, map_location=device))

    model_list.append(models.efficientnet_b1(weights=None))
    model_list[3].classifier[1] = torch.nn.Linear(1280, 21)
    model_list[3].load_state_dict(torch.load("bestyolob1pretrain", weights_only=True, map_location=device))
    
    model_list.append(models.efficientnet_b2(weights=None))
    model_list[4].classifier[1] = torch.nn.Linear(1408, 21)
    model_list[4].load_state_dict(torch.load("Model2_Epochs20_PretraindTrue_best_temp_weights", map_location=device))

    model_list.append(models.efficientnet_b2(weights=None))
    model_list[5].classifier[1] = torch.nn.Linear(1408, 21)
    model_list[5].load_state_dict(torch.load("bestyolob2pretrain", map_location=device))

    model_list.append(models.efficientnet_b3(weights=None))
    model_list[6].classifier[1] = torch.nn.Linear(1536, 21)
    model_list[6].load_state_dict(torch.load("Model3_Epochs20_PretraindTrue_best_temp_weights", weights_only=True, map_location=device))

    model_list.append(models.efficientnet_b3(weights=None))
    model_list[7].classifier[1] = torch.nn.Linear(1536, 21)
    model_list[7].load_state_dict(torch.load("bestyolob3pretrain", weights_only=True, map_location=device))

    for model in model_list:
        model.eval()

    return model_list

def create_ensemble_weights(model_list):
    """Used to find optimal combination weights for an ensemble of the input list.
    Finds weights by analysing the accuracy of each output label for each model, then normalising them among the models.
    Args: A list of models in eval mode
    Returns: A list of weights of length len(model_list)x21 
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = np.zeros((len(model_list), 21), dtype=float)
    accuracies = np.zeros((len(model_list), 21))
    counts = np.zeros((len(model_list), 21))
    validation_dataset = preprocessing.get_validation_dataloader().dataset
    
    for i in range(len(model_list)):
        for pair in validation_dataset:
            model_guess = model_list[i](pair[0].unsqueeze(0))
            model_guess = model_guess.argmax(dim=1).item()
            accuracies[i][model_guess] += int(pair[1] == model_guess)
            counts[i][model_guess] += 1

        for j in range(21):
            weights[i][j] = accuracies[i][j]/counts[i][j]

    weights = weights.transpose()

    for i in range(21):
        print(sum(weights[i]))
        weights[i] /= sum(weights[i])

    weights = weights.transpose()

    np.save("Ensemble weights", weights)

    return weights