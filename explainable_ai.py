import torch
from captum.attr import Occlusion
import matplotlib.pyplot as plt
import numpy as np
import preprocessing 
import model_creator
import torchvision.models as models



data = preprocessing.get_one_dataloader(shuffled=False, image_side_length=224, augment_factor=0,batch_size=1)

heatmaps = []
true_labels = []
predicted_labels = []

#important if u dont have a GPU u need to set GPU = false at line 46 in model.py
#to use it tip in 
#python
#import explainable_ai
#explainable_ai.getocclusion(window=40, stride=16, picuture_index=0)
# you need to close the window to get the next image
#when finnished use exit() to close the console


#man ruft getocclusion auf und übergibt die parameter die man haben möchte
#window = wie groß das patch sein soll hier immer Quadratisch
#stride = schritteweite


def getocclusion( window, stride, picuture_index):
    global heatmaps, true_labels, predicted_labels
    heatmaps = []
    true_labels = []
    predicted_labels = []
   
    model = model_creator.load_model()
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

    picture = heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(picture)
    plt.colorbar()
    plt.title(f"Heatmap number: {picuture_index}, true class: {true_class}, predicted class: {predicted_class} ")
    plt.show()

    return 





