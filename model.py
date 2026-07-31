import model_creator

def train_test_models():
    """Trains multiple models to remove the need for manual queuing
    """
    file = open("accuracy_list.txt", 'a')
    model, validation = model_creator.train_model(use_gpu=True, epochs=20, model_number=4, create_validation_dataloader=False, augment=True, batch_size=32, pretrained=True)
    accuracy = model_creator.evaluate_model(model, validation)
    file.write("Yolo Simple CNN " + str(accuracy) + "\n")
    del(model)
    del(validation)
    file.close()

    file = open("accuracy_list.txt", 'a')
    model, validation = model_creator.train_model(use_gpu=True, epochs=20, model_number=1, create_validation_dataloader=False, augment=True, batch_size=16, pretrained=True)
    accuracy = model_creator.evaluate_model(model, validation)
    file.write("Yolo EfficientNet_B1 " + str(accuracy) + "\n")
    del(model)
    del(validation)
    file.close()

    file = open("accuracy_list.txt", 'a')
    model, validation = model_creator.train_model(use_gpu=True, epochs=20, model_number=0, create_validation_dataloader=False, augment=True, batch_size=16, pretrained=True)
    accuracy = model_creator.evaluate_model(model, validation)
    file.write("Yolo EfficientNet_B0 with default/pretrained weights " + str(accuracy) + "\n")
    del(model)
    del(validation)
    file.close()

    file = open("accuracy_list.txt", 'a')
    model, validation = model_creator.train_model(use_gpu=True, epochs=20, model_number=1, create_validation_dataloader=False, augment=True, batch_size=16, pretrained=True)
    accuracy = model_creator.evaluate_model(model, validation)
    file.write("Yolo EfficientNet_B1 with default/pretrained weights " + str(accuracy) + "\n")
    del(model)
    del(validation)
    file.close()

    file = open("accuracy_list.txt", 'a')
    model, validation = model_creator.train_model(use_gpu=True, epochs=20, model_number=2, create_validation_dataloader=False, augment=True, batch_size=16, pretrained=True)
    accuracy = model_creator.evaluate_model(model, validation)
    file.write("Yolo EfficientNet_B2 with default/pretrained weights " + str(accuracy) + "\n")
    del(model)
    del(validation)
    file.close()

    file = open("accuracy_list.txt", 'a')
    model, validation = model_creator.train_model(use_gpu=True, epochs=20, model_number=3, create_validation_dataloader=False, augment=True, batch_size=16, pretrained=True)
    accuracy = model_creator.evaluate_model(model, validation)
    file.write("Yolo EfficientNet_B3 with default/pretrained weights " + str(accuracy) + "\n")
    del(model)
    del(validation)
    file.close()

train_test_models()
