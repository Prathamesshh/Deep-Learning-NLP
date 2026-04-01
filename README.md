## Project Description: Next Word Prediction Using LSTM
#### Project Overview:

This project aims to develop a deep learning model for predicting the next word in a given sequence of words. The model is built using Long Short-Term Memory (LSTM) networks, which are well-suited for sequence prediction tasks. The project includes the following steps:

1- Data Collection: We use the text of Shakespeare's "Hamlet" as our dataset. This rich, complex text provides a good challenge for our model.

2- Data Preprocessing: The text data is tokenized, converted into sequences, and padded to ensure uniform input lengths. The sequences are then split into training and testing sets.

3- Model Building: An LSTM model is constructed with an embedding layer, two LSTM layers, and a dense output layer with a softmax activation function to predict the probability of the next word.

4- Model Training: The model is trained using the prepared sequences, with early stopping implemented to prevent overfitting. Early stopping monitors the validation loss and stops training when the loss stops improving.

5- Model Evaluation: The model is evaluated using a set of example sentences to test its ability to predict the next word accurately.

6- Deployment: A Streamlit web application is developed to allow users to input a sequence of words and get the predicted next word in real-time.

#### Key Features
Text preprocessing and tokenization
Sequence generation for language modeling
Implementation of LSTM and GRU architectures
Training deep learning models for next word prediction
Model comparison between LSTM and GRU
Prediction interface using a Python application
Visualization of model performance


#### Technologies Used
Python
TensorFlow / Keras
Natural Language Processing (NLP)
Jupyter Notebook
NumPy & Pandas


#### Dataset

The model is trained on the text file:
hamlet.txt
This dataset contains text from Shakespeare's Hamlet, allowing the model to learn contextual word relationships.


#### Project Structure
Deep-Learning-NLP/
│
├── README.md                     # Project documentation     
├── app.py                        # Prediction application     
├── experiments.ipynb             # Model training notebook    
├── hamlet.txt                    # Training dataset    
│      
├── lstm_model.h5                 # Trained LSTM model    
├── gru_model.h5                  # Trained GRU model     
├── next_word_lstm.h5             # Final prediction model     
│      
├── model_comparison.png          # Visualization of results   
├── model_comparison_results.json # Model evaluation results     


#### Model Architecture
LSTM Model

LSTM networks are designed to capture long-term dependencies in sequential data, making them suitable for natural language tasks.

GRU Model

GRU is a simplified variant of LSTM that provides faster training while maintaining good performance.

Both models were trained and evaluated to compare their prediction capabilities.

#### How It Works
Load the text dataset
Preprocess the text (tokenization and cleaning)
Convert sentences into sequences
Train LSTM and GRU models
Save trained models
Use the trained model to predict the next word

Example:
Input:
To be or not to
Prediction:
be
