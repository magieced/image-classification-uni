import model_creator

# Used to create multiple models using different architectures to assess architecture accuracy
def train_test_models():
    model0, validation0 = model_creator.train_model(epochs=20, model_number=0,create_validation_dataloader=True)
    model1, validation1 = model_creator.train_model(epochs=20, model_number=1,create_validation_dataloader=True)
    model2, validation2 = model_creator.train_model(epochs=20, model_number=2,create_validation_dataloader=True)
    model3, validation3 = model_creator.train_model(epochs=20, model_number=3,create_validation_dataloader=True)
    model4, validation4 = model_creator.train_model(epochs=20, model_number=4,create_validation_dataloader=True)

    accuracy0 = model_creator.evaluate_model(model0, validation0)
    accuracy1 = model_creator.evaluate_model(model1, validation1)
    accuracy2 = model_creator.evaluate_model(model2, validation2)
    accuracy3 = model_creator.evaluate_model(model3, validation3)
    accuracy4 = model_creator.evaluate_model(model4, validation4)

    file = open("accuracy_list.txt", 'w')
    file.write("b0 accuracy: " + str(accuracy0) + "\n")
    file.write("b1 accuracy: " + str(accuracy1) + "\n")
    file.write("b2 accuracy: " + str(accuracy2) + "\n")
    file.write("b3 accuracy: " + str(accuracy3) + "\n")
    file.write("Simple CNN accuracy: " + str(accuracy4) + "\n")
    file.close()
train_test_models()
