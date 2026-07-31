import random
import torch
from captum.attr import Occlusion
import matplotlib.pyplot as plt
import preprocessing 
import model_creator

model = model_creator.load_model()
device = next(model.parameters()).device
occlusion = Occlusion(model)

def load_dataloader():
    return preprocessing.get_one_dataloader(shuffled=False, image_side_length=224, augment_factor=0, batch_size=1)

my_dataloader = load_dataloader() 


def getocclusion(window: int, stride: int, picture_index: int):
    dataset = my_dataloader.dataset
    input_tensor, label_tensor = dataset[picture_index]
    input_tensor = input_tensor.unsqueeze(0)


    true_class = int(label_tensor)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        out = model(input_tensor)
        predicted_class = torch.argmax(out, dim=1).item()

    sliding_window_shapes = (3, window, window)
    strides = (3, stride, stride)

    attribute = occlusion.attribute(
        input_tensor,
        target=predicted_class, 
        sliding_window_shapes=sliding_window_shapes, 
        strides=strides 
    )

    heatmap = attribute.squeeze(0).cpu().detach().numpy().mean(axis=0)

    original_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title(f"Image Index: {picture_index} (True class: {true_class}) (Predicted class: {predicted_class})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    image = plt.imshow(heatmap)
    plt.title(f"Heatmap ")
    plt.axis('off')
    plt.colorbar(image)

    plt.tight_layout()
    plt.show()

    return heatmap, true_class, predicted_class