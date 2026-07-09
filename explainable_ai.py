import torch
from captum.attr import Occlusion
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import dataloaders
from model import #model

training_data, validation_data = dataloaders(shuffle = False)

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") wurde empfohlen um das berechen zu beschleunigen 

model = #model instanzieren 
#model to device
model.eval()#notwendig für konstante Ergbnisse 

occlusion = Occlusion(model)

heatmaps = []
true_labels = []
predicted_labels = []

for input_tensor, i in validation_data:
    #input_tensor= input_tensor.to(device)
    true_class = i.item()

    with torch.no_grad():
        out = model(input_tensor)
        predicted_class = torch.argmax(out, dim = 1).item()

    attribute = occlusion.attribute(
        input_tensor,
        target = predicted_class, #muss evtl noch angepasst werden für jede klasse
        sliding_window_shapes = (3,15,15), #bestimmt die größe des abgedeckten Bereichs
        strides = (3,8,8) #schritt weite je kleiner desto präziser bnötigt aber mehr rechenkapazität

    )

    heatmap = attribute.squeeze(0).detach().numpy().mean(axis= 0)
    #heatmap = attribute.squeeze(0).cpu().detach().numpy().mean(axis= 0)#wenn über gpu


    heatmaps.append(heatmap)
    true_labels.append(true_class)
    predicted_labels.append(predicted_class)

def display_picture(number):
    picture = heatmaps[number]
    plt.figure(figsize=(5,5))
    plt.imshow(picture)
    plt.colorbar()
    plt.title(f"Heatmap number: {number}")
    plt.show()


