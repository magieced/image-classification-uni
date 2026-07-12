import model_creator

# Used to create multiple models using different architectures to assess architecture accuracy
def train_test_models():
    file = open("accuracy_list.txt", 'w')

    model0, validation0 = model_creator.train_model(use_gpu=True, epochs=20, model_number=0,create_validation_dataloader=True)
    accuracy0 = model_creator.evaluate_model(model0, validation0)
    file.write("b0 accuracy: " + str(accuracy0) + "\n")
    del(model0)
    del(validation0)

    model1, validation1 = model_creator.train_model(use_gpu=True, epochs=20, model_number=1,create_validation_dataloader=True)
    accuracy1 = model_creator.evaluate_model(model1, validation1)
    file.write("b1 accuracy: " + str(accuracy1) + "\n")
    del(model1)
    del(validation1)

    model2, validation2 = model_creator.train_model(use_gpu=True, epochs=20, model_number=2,create_validation_dataloader=True)
    accuracy2 = model_creator.evaluate_model(model2, validation2)
    file.write("b2 accuracy: " + str(accuracy2) + "\n")
    del(model2)
    del(validation2)

    model3, validation3 = model_creator.train_model(use_gpu=True, epochs=20, model_number=3,create_validation_dataloader=True)
    accuracy3 = model_creator.evaluate_model(model3, validation3)
    file.write("b3 accuracy: " + str(accuracy3) + "\n")
    del(model3)
    del(validation3)

    model4, validation4 = model_creator.train_model(use_gpu=True, epochs=20, model_number=4,create_validation_dataloader=True)
    accuracy4 = model_creator.evaluate_model(model4, validation4)
    file.write("Simple CNN accuracy: " + str(accuracy4) + "\n")
    del(model4)
    del(validation4)

    model5, validation5 = model_creator.train_model(use_gpu=True, epochs=20, model_number=4, create_validation_dataloader=True, augment_factor=0)
    accuracy5 = model_creator.evaluate_model(model5, validation5)
    file.write("Simple CNN accuracy with 0 augmentation_factor: " + str(accuracy5) + "\n")
    del(model5)
    del(validation5)

    model6, validation6 = model_creator.train_model(use_gpu=True, epochs=20, model_number=4,create_validation_dataloader=True, augment_factor=1)
    accuracy6 = model_creator.evaluate_model(model6, validation6)
    file.write("Simple CNN accuracy with 1 augmentation_factor: " + str(accuracy6) + "\n")
    del(model6)
    del(validation6)

    file.close()

train_test_models()
