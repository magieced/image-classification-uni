import torch
from captum.attr import Occlusion
import matplotlib.pyplot as plt
import numpy as np
import preprocessing 
import model_creator
import torchvision.models as models


data = preprocessing.get_one_dataloader(shuffled=False, image_side_length=224, augment_factor=0)

heatmaps = []
true_labels = []
predicted_labels = []

def getocclusion(model_number, weigth_number, window, stride, picuture_index):
    global heatmaps, true_labels, predicted_labels
    heatmaps = []
    true_labels = []
    predicted_labels = []
   
    model = model_creator.load_model(model_number, weigth_number)
    occlusion = Occlusion(model)

    sliding_window_shapes1 = (3, window, window)
    strides1 = (3, stride, stride)

    data_iterator = iter(data)

    for _ in range(picuture_index):
        try: 
            next(data_iterator)
        except StopIteration:
            print(f"Fehler: Index {picuture_index} ist zu groß!")
            return

    try: 
        input_tensor, i = next(data_iterator)
    except StopIteration:
        print(f"Fehler: Index {picuture_index} außerhalb der Reichweite!")
        return
        
    true_class = i.item()
    
    with torch.no_grad():
        out = model(input_tensor)
        predicted_class = torch.argmax(out, dim=1).item()

    attribute = occlusion.attribute(
        input_tensor,
        target=predicted_class, 
        sliding_window_shapes=sliding_window_shapes1, 
        strides=strides1 
    )

    heatmap = attribute.squeeze(0).detach().numpy().mean(axis=0)

    heatmaps.append(heatmap)
    true_labels.append(true_class)
    predicted_labels.append(predicted_class)

    picture = heatmaps[0]
    plt.figure(figsize=(5, 5))
    plt.imshow(picture)
    plt.colorbar()
    plt.title(f"Heatmap number: {picuture_index}, true class: {true_class}, predicted class: {predicted_class} ")
    plt.show()

    return 

