# sign-language-translation
This was our chosen project for ICHack 2022.

It is a sign language translator written in Python. When main.py is run, the user can repeatedly capture photos of letters using the GUI that pops up, which launches the front-facing camera and places a green square on the screen to help guide the user's hand. When the user is finished, they can simply close the camera and the program will then feed the photos to a pre-trained model, trained in model_training.py and stored in cnn.h5. The command-line then displays the alphabetical translation of the user's sign language input.

The backend was written using Tensorflow, Numpy, PIL and Pandas and the front-end was written primarily using OpenCV. 
Please note that the csv file containing the training data was not uploaded due to size restrictions, but is of a similar nature to the test data (sign_mnist_test.csv)
