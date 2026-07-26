# image-classification-uni
Group Project for University, classifying an image into one of 21 classes

All that needs to be done to run the model is running inference.py

If manual setup of the model is necessary, it can be done in the following way:
1. Preprocessing:
    a) use the preprocessing.single_im_preprocessing method on a PIL Image
    b) unsqueeze(0) the resulting tensor
2. Model:
    a) run model_creator.load_model() and save the output as the model
    b) forward the tensor gotten from step 1 into the model
    c) get the maximum confidence in the returned logit, for example with argmax(dim=1).item()
    d) convert class 20 to class -1
As a result, you will have an int in range [-1, 19], corresponding to the classes in inference.py

Training was done on images gotten from the sources listed in 21ClassDataset\CSV_creator.py
You can train a model by running model_creator.train_model(), with preprocessing being completed automatically in the train_model function.