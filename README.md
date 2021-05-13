# Smart-Music-Player
Real time music streaming using emotion and age
Emotion detection:
Emotion_images.pkl : images are read and dumped in a file with the help pf pickle so that we don’t have to read image files again and again we just load the pickle file .
Emotion_label.pkl : Similarly labels are also dumped
Various notebooks are provided in the pdf format which shows all the tried combinations in Convolution neural network with classification report, confusion matrix and other plots.
Grid search with cross validation: A script is written in python which implements grid search with cross validation and outputs the best combination of the parameters to be used to avoid over fitting.
Age detection:
Age_images.pkl : images are read and dumped in a file with the help pf pickle so that we don’t have to read image files again and again we just load the pickle file .
Age_list.pkl: Similarly, labels are also dumped
Notebook is provided in the pdf format which shows some of the tried combinations in Convolution neural network with classification report, confusion matrix and other plots.
Grid search with cross validation: A script is written in python which implements grid search with cross validation and outputs the best combination of the parameters to be used to avoid over fitting.

Web application
Application.py : opens the webcam
Predict.py :  Face is detected with the help of cascade classifier and image is cropped.
Saved model is loaded with the help of pickle and cropped image is fed to it to return the emotion and age.
Sqldb.py : Get the list of all the songs that belongs to particular emotion from the hashmap. Get the list of all the songs that belongs to Age group . find the intersection of both the lists and append the resultant to the queue of songs to be played.
Songs.py : List of songs and their information. Small set of songs uploaded on google drive.


