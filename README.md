# EchoMood-_Decoding-Emotions-From-Audio
EchoMood is an advanced audio analysis tool designed to detect and interpret human emotions through voice. By analyzing soundwaves, it decodes subtle vocal cues to accurately identify emotional states, bringing a deeper understanding to how we express and perceive feelings through sound.


# Model Training
Detect emotions from audio using a Multilayer Perceptron (MLP) classifier from sklearn.neural_network. The project utilizes feature extraction techniques combined with grid search for hyperparameter tuning to improve model performance.

Steps:
<dl><b>Feature Extraction:</b>

<li>A custom extract_feature() function extracts key audio features, including MFCCs, chroma, and mel spectrograms using librosa. These features serve as the input for training the model. The RAVDESS dataset, consisting of audio files labeled with different emotions, is used for this purpose.
The function processes the audio files, transforming them into a feature matrix, with a subset of emotions ('calm', 'happy', 'fearful', and 'disgust') used for training the model.</li></dl>
<dl><b>Data Preparation:</b>

<li>The dataset is split into training and testing sets with a 75-25 ratio. The features are stored in a matrix, where each row represents a different audio sample, and each column represents an extracted feature (180 features in total).</li></dl>
<dl><b>MLP Classifier & Hyperparameter Tuning:</b>

<li>An MLPClassifier model is trained using the training set with a hyperparameter search performed using GridSearchCV. The hyperparameter grid includes various configurations of hidden layer sizes, activation functions (tanh, relu), solvers (adam, sgd), learning rates (constant, adaptive), batch sizes, and iteration limits.
The grid search evaluates 648 different combinations, identifying the best parameters: hidden layers (200, 200), activation tanh, solver adam, batch size 256, alpha 0.01, and learning rate constant.</li>
<li>The best cross-validation accuracy achieved is 78.3%.</li></dl>

<dl><b>Model Training & Evaluation:</b>

<li>The MLP model with the best parameters is trained on the entire training set. It achieves a training accuracy of 94.44%, and a test accuracy of 73.44%, showing reasonable performance on unseen data.
A learning curve plot visualizes the model's loss values over iterations, giving insight into the model's training progress.</li></dl>
<dl><b>Model Deployment:</b>

<li>The optimized model is saved using pickle and deployed within a Django application. In the Django web app, users can upload audio files, and the saved model predicts the emotion conveyed in the audio. The app also displays GIFs corresponding to the detected emotions for an enhanced user experience.
</li></dl>
<p>The notebook demonstrates a complete pipeline for emotion detection from audio, showcasing both feature extraction and the implementation of grid search for hyperparameter tuning of a neural network model.</p>


# Output
<h3>Predict Emotion</h3> 

![](https://github.com/Rashid9226/EchoMood-_Decoding-Emotions-From-Audio/blob/main/imgs/predict.PNG)
<br>
<h3>result</h3>  

![](https://github.com/Rashid9226/EchoMood-_Decoding-Emotions-From-Audio/blob/main/imgs/result.PNG)
