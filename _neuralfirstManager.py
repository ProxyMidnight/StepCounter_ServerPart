from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
from Managers._fileManager import get_data_from_file
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.regularizers import l1, l2 # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from math import sqrt

def label_to_numeric(label):
    """
    Converts a label to its corresponding numeric value.

    Args:
        label (str): The label to be converted.

    Possible Inputs:
        'sitting', 'standing', 'stepping'

    Returns:
        int: The numeric value corresponding to the label.

    Possible Outputs:
        0 (for 'sitting'), 1 (for 'standing'), 2 (for 'stepping')
    """
    label_to_numeric = {
        'sitting': 0,
        'standing': 1,
        'stepping': 2
    }
    return label_to_numeric[label]

def numeric_to_label(numeric):
    """
    Converts a numeric value to its corresponding label.

    Args:
        numeric (int): The numeric value to be converted.

    Possible Inputs:
        0, 1, 2

    Returns:
        str: The label corresponding to the numeric value.

    Possible Outputs:
        'sitting' (for 0), 'standing' (for 1), 'stepping' (for 2)
    """
    numeric_to_label = {
        0: 'sitting',
        1: 'standing',
        2: 'stepping'
    }
    return numeric_to_label[numeric]

def parse_datetime(datetime_str):
    """
    Parses a string representation of a datetime object and returns the corresponding datetime object.

    Args:
        datetime_str (str): A string representing a datetime in the format '%Y-%m-%d %H:%M:%S.%f'.

    Returns:
        datetime.datetime: The parsed datetime object.

    Raises:
        ValueError: If the input string is not in the correct format.

    """
    date_time_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
    return date_time_obj

def filter_and_average_accelerometer_data(sensor_data):
    """
    Filters and averages the accelerometer data from the given sensor data.

    Args:
        sensor_data (list): A list of sensor data points.

    Returns:
        list: A list of averaged accelerometer data points.
    """
    accel_data = [data_point for data_point in sensor_data if data_point[1] == 'ACCELEROMETER']

    accel_data_datatime = [[ parse_datetime(datetime),float(x), float(y), float(z)] for (datetime, _, x, y, z) in accel_data]

    i = 0
    averaged_data = []
    start_time = accel_data_datatime[0][0]
    while i < len(accel_data_datatime):
        current_values = []
        end_time = start_time + timedelta(milliseconds=100)

        while i < len(accel_data_datatime) and accel_data_datatime[i][0] < end_time:
            current_values.append(accel_data_datatime[i][1:])
            i += 1

        if current_values:
            mean_values = np.mean(current_values, axis=0).tolist()
            averaged_data.append(mean_values)                
        start_time = end_time
    
    # Pad the averaged_data list with zeros if it has less than 20 records
    while len(averaged_data) < 20:
        averaged_data.append([0.0, 0.0, 0.0])

    return averaged_data

class FirstNeuralNetwork:
    """
    A class representing a neural network model for activity recognition: sitting, standing, stepping.

    Attributes:
        model: The neural network model.
    """

    def __init__(self, model_path_to_load_model=None):
        """
        Initializes a new instance of the FirstNeuralNetwork class.

        Args:
            model_path (str, optional): The path to a pre-trained model. Defaults to None.
        """
        self.model = None
        if model_path_to_load_model:
            self.load_model(model_path_to_load_model)
        else:
            # 3 stands for 3 input instances: x, y, z
            self.model = self.build_model((20, 3))
    
    def build_model(self, input_shape):
        """
        Builds the neural network model.

        Args:
            input_shape (tuple): The shape of the input data.

        Returns:
            The built neural network model.
        """
        # Define a Sequential model architecture
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),  # LSTM layer with 64 units, expects input shape as defined, returns sequences to the next layer
            Dropout(0.5),  # Dropout layer to prevent overfitting, dropping 50% of the units' activations randomly
            LSTM(32),  # Second LSTM layer with 32 units, does not return sequences (returns only the last output)
            Dropout(0.5),  # Another Dropout layer, also dropping 50% of the units' activations
            Dense(3, activation='softmax', kernel_regularizer=l2(0.01))  # Dense output layer with 3 units, softmax activation for multi-class classification, and L2 regularization
        ])

        # Compile the model to configure the learning process
        model.compile(
            optimizer='adam',  # Optimizer 'adam' - an algorithm for first-order gradient-based optimization of stochastic objective functions
            loss='categorical_crossentropy',  # Loss function for multi-class classification problems
            metrics=['accuracy']  # List of metrics to be evaluated by the model during training and testing; here 'accuracy' to judge performance
        )
        return model  # Return the compiled model

    def load_and_prepare_study_data(self, file_paths):
        """
        Loads and prepares the study data for training the neural network.

        Args:
            file_paths (list): A list of file paths containing the study data.

        Returns:
            The preprocessed input data (X) and the corresponding labels (y).
        """
        all_accel_data = []
        all_labels = []

        for file_path in file_paths:
            activity_label, sensor_data = get_data_from_file(file_path)  

            averaged_data = filter_and_average_accelerometer_data(sensor_data)

            all_accel_data.append(averaged_data)
            all_labels.append(label_to_numeric(activity_label))  

        # I don't know why but it's necessary
        X = pad_sequences(all_accel_data)

        # Apply one-hot encoding to the numeric labels
        y_one_hot = to_categorical(all_labels, num_classes=3)

        return X, y_one_hot

    def teach_neural_network(self, file_paths_to_study_data):
        """
        Trains the neural network model using the provided study data.

        Args:
            file_paths (list): A list of file paths containing the study data.
        """
        # x - represents the input data. For example: 
        # y - represents the target output. For example: 
        x, y = self.load_and_prepare_study_data(file_paths_to_study_data)
        
        # early_stopping is a callback to stop training when a monitored metric has stopped improving.
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Metric to be monitored; 'val_loss' monitors the validation loss.
            patience=10,         # Number of epochs with no improvement after which training will be stopped.
            mode='min',          # The direction is 'min', meaning training will stop when the quantity monitored has stopped decreasing.
            restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity.
        )

        # Training the model. Here 'x' represents preprocessed input data, and 'y' are the target labels encoded in one-hot format.
        self.model.fit(
            x,  # Input features to the model.
            y,  # Target output or labels to the model.
            epochs=5000,  # Maximum number of epochs to try training for.
            validation_split=0.2,  # Fraction of the data to be used as validation data. 20% of the data will be used to validate the model's performance.
            callbacks=[early_stopping]  # List of callbacks to apply during training. In this case, early stopping to prevent overfitting.
        )

    def predict(self, x):
        """
        Predicts the output for the given input using the trained neural network model.

        Args:
            x (numpy.ndarray): The input data for prediction.

        Returns:
            numpy.ndarray: The predicted output. Return list of list of possibilities of every class. For example: [0.0034027  0.99336857 0.00322873].
        """
        prediction = self.model.predict(x)
        return prediction[0]
        
    def save_model(self, file_path_to_save_model):
        """
        Saves the trained neural network model to a file. Saves in h5 format.

        Args:
            file_path_to_save_model (str): The path to save the model.

        """
        self.model.save(file_path_to_save_model)

    def load_model(self, file_path_to_load_model):
        """
        Loads a pre-trained neural network model from a file. Loads in h5 format.

        Args:
            file_path_to_load_model (str): The path to the pre-trained model.

        """
        self.model = load_model(file_path_to_load_model)

