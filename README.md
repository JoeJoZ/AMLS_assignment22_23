# AMLS_assignment

In this project, four distinct challenging scopes are addressed under the supervised machine learning paradigm. They comprise binary classification tasks for gender (A1) and smile detection (A2) along with multi-categorical classification tasks concerning eye-colour (B2) and face-shape (B1) recognition.

## Required Packages

- PyTorch
- OpenCV-python
- scikit-learn
- dlib
- Numpy
- Pandas
- timm
- scipy
- imutils
- scikit-image
- tqdm

## How to start

This project uses Pycharm for development. You can open this project with Pycharm and run main.py

## Role of each file

**main.py** is the starting point of the entire project. It defines the order in which instructions are realised. More precisely, it is responsible to call functions from other files to test all models.

Each folder **A1**, **A2**, **B1**, **B2** contains the python scripts specific to that task. **train.py** contains the training code for the model. **inference.py** contains the test code for the model. **models.py** contains the model used for that task. **FaceDataset.py** defines the method of reading the dataset. **config.py** contains code used to set hyperparameters of the models.

The folder **utils** contains some utility classes. For example, for saving checkpoints and for printing logs code.

The folder **outputs** is used to save the trained model files.

**shape_predictor_68_face_landmarks.dat** is the pre-trained face landmark extractor required for task A2.

Datasets must be placed in the **Datasets** folder in the root directory as required

Because the models in output file is too big, I up load the model to Google Drive. Please down load these models and put them into correct files.

https://drive.google.com/file/d/1wRXSYoI_YwxsR8nYiXmBBvoFGYvNbrqs/view?usp=sharing  This model should been put into 'outputs''12-15-A1-lr0.001-bs32'

https://drive.google.com/file/d/1mlAdcMZ-a9Y5KLcVygOBnk4aYpd1UrN1/view?usp=sharing  This model should been put into 'outputs''12-23-A2-atten1-lr0.001-bs32'
