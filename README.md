# image-classification-uni
Group Project for University, classifying an image into one of 21 classes

To set up the environment, create a new venv environment and use pip install requirements.txt, which contains all needed packages.

To run the model, run inference.py. inference.py automatically loads the model and does all necessary preprocessing.
This, to our knowledge, creates the best output.

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

If you want to create an ensemble model, create multiple versions of each EfficientNet pretrained model and store them in load_multiple_models as necessary, then pass them to the create_ensemble_weights function.

Training was done on images gotten from the sources listed in 21ClassDataset\CSV_creator.py
You can train a model by running model_creator.train_model(), with preprocessing being completed automatically in the train_model function.
To get the accuracy of the model on a validation subset of the training set, run train_model with create_validation_dataloader=True and pass the two outputs into model_creator.evaluate_model in the same order as they were returned.