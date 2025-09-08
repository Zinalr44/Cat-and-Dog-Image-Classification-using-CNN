# Cat-and-Dog-Image-Classification-using-CNN
This repository contains a deep learning project for classifying images of cats and dogs using Convolutional Neural Networks (CNNs) with Keras and TensorFlow. The project involves data preprocessing, model training, evaluation, and visualization of results.

Key Features:
Data Preprocessing: Organizes and standardizes image data for training and validation.
CNN Model: Builds a CNN model for binary classification.
Training and Evaluation: Trains the model and evaluates its performance on the validation set.
Visualization: Plots training history and visualizes predictions.
Requirements:
Python 3.x
pandas
numpy
matplotlib
tqdm
OpenCV
scikit-learn
TensorFlow
Keras
seaborn
Files:
train: Directory containing training images of cats and dogs.
test1: Directory containing test images.
trainvalidfull4keras: Directory structure for training and validation sets.
Usage:
Load Dataset: Ensure the dataset is organized with separate directories for training and validation images.
Preprocess Data: Use the provided script to preprocess and organize the data.
Train the Model: Train the CNN model using the training data.
Evaluate and Visualize: Evaluate the model on the validation data and visualize the results.

## Features
- Data loading and preprocessing from directories
- Train-validation split with data augmentation
- CNN model architecture with convolutional and pooling layers
- Model training with callbacks for early stopping and learning rate reduction
- Evaluation metrics including accuracy, loss, classification report, and confusion matrix
- Visualization of training history, predictions, and sample images
- Model saving for future use

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV (cv2)
- tqdm
- Seaborn (for confusion matrix visualization)

Install dependencies using:
```
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python tqdm seaborn
```

## Installation
1. Clone or download the repository.
2. Ensure the dataset is placed in the specified directories (e.g., `C:\Users\Dell.com\Desktop\ml\project 4\train` and `C:\Users\Dell.com\Desktop\ml\project 4\test1`). Update paths in the notebook if necessary.
3. Open the `cat_dog.ipynb` notebook in Jupyter Notebook or JupyterLab.
4. Run the cells sequentially to execute the code.

## Usage
1. Load the notebook in Jupyter.
2. Execute the cells to:
   - Import libraries and set constants.
   - Load and visualize dataset statistics.
   - Split data into training and validation sets.
   - Build and compile the CNN model.
   - Train the model with data generators.
   - Evaluate and visualize results.
3. The trained model will be saved as `cats_and_dogs_classifier.h5`.
4. Use the model for predictions on new images by loading and preprocessing them similarly.

## Model Details
- **Architecture**: Sequential CNN with 3 convolutional layers, max pooling, flattening, dense layers, and dropout for regularization.
- **Input Shape**: 224x224x3 (RGB images resized to 224x224).
- **Output**: Binary classification (sigmoid activation).
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Training**: 5 epochs with batch size 32, data augmentation (shear, zoom, horizontal flip).

## Results
- Validation accuracy achieved: [Insert actual accuracy from notebook, e.g., ~85-90% based on typical runs]
- Includes classification report and confusion matrix for detailed evaluation.
- Visualizations for training/validation accuracy/loss curves and sample predictions.

## Code Review Summary
- **Strengths**: Comprehensive pipeline from data prep to evaluation. Good use of data augmentation and visualization.
- **Improvements**:
  - Add more comments for clarity.
  - Use relative paths or environment variables for directories to improve portability.
  - Handle potential errors (e.g., missing directories).
  - Avoid redundant code (e.g., model definition appears twice).
  - Consider using a configuration file for hyperparameters.
  - Add unit tests or validation for data integrity.

## Contributing
Feel free to fork the repository and submit pull requests for improvements.

## License
This project is open-source and available under the MIT License.
