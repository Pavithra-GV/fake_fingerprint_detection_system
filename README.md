# fake fingerprint detection system
replace the content in "< >" with the path
---

In the realm of biometric security, fingerprint recognition systems are widely adopted for their convenience and reliability. However, the growing sophistication of spoofing techniques, where fake fingerprints are used to deceive these systems, poses a significant threat. This necessitates the development of robust methods to distinguish between genuine and counterfeit fingerprints. The provided code serves this purpose by leveraging deep learning techniques to train a Convolutional Neural Network (CNN) capable of identifying fake fingerprints with high accuracy. By using a carefully designed model architecture and a comprehensive dataset, this system enhances the security of fingerprint recognition systems, making them more resilient to fraudulent attempts. Additionally, the Gradio interface facilitates easy interaction with the model, allowing users to quickly verify the authenticity of fingerprints through a user-friendly platform.
---
The `FakeFingerprintDetection.py` is designed to detect fake fingerprints using a convolutional neural network (CNN). Here’s a detailed breakdown of what each section of the code does:

### Imports

```python
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import load_model
import cv2
```
- Standard libraries (`os`, `numpy`, `cv2`) and TensorFlow/Keras libraries are imported for building and training the neural network, as well as for image processing and visualization.

### Load Data

```python
data_dir = 'train'
data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(256, 256), batch_size=32)
```
- Loads the dataset from a directory named 'train', resizes the images to 256x256 pixels, and batches them with a size of 32.

### Visualize Some Images

```python
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
```
- Takes a batch of images from the dataset and visualizes four of them to get an idea of what the data looks like.

### Data Preprocessing

```python
data = data.map(lambda x , y: (x/255 , y))
```
- Normalizes the image data by scaling the pixel values to the range [0, 1].

### Calculate Dataset Sizes and Split Data

```python
data_size = tf.data.experimental.cardinality(data).numpy()
train_size = int(data_size * 0.7)
val_size = int(data_size * 0.15)
test_size = data_size - train_size - val_size

if train_size == 0:
    train_size = 1
    val_size = max(1, data_size - 2)
    test_size = data_size - train_size - val_size

if val_size == 0:
    val_size = 1
    test_size = max(1, data_size - train_size - 1)

if test_size == 0:
    test_size = 1
    val_size = max(1, data_size - train_size - 1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
```
- Calculates the sizes for the training, validation, and test sets. It ensures that the splits are non-zero, then splits the data accordingly.

### Model Building

```python
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
```
- Defines a Sequential model with three convolutional layers, each followed by a max pooling layer, then flattens the output and adds two dense layers. The last layer has a sigmoid activation function for binary classification. The model is compiled with the Adam optimizer and binary crossentropy loss.

### Train Model

```python
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
```
- Sets up a TensorBoard callback for logging, then trains the model for 20 epochs using the training and validation data.

### Plot Performance

```python
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
```
- Plots the training and validation loss and accuracy over the epochs to visualize the performance of the model during training.

### Evaluate

```python
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
```
- Evaluates the model on the test data using precision, recall, and accuracy metrics.

### Test the Model

```python
img = cv2.imread('<insert test image file path>')
resize = tf.image.resize(img, (256, 256))

yhat = model.predict(np.expand_dims(resize / 255, 0))

if yhat > 0.5:
    print('Predicted as real')
else:
    print('Predicted as fake')
```
- Loads a test image, resizes it, and uses the trained model to predict whether the fingerprint is real or fake.

### Save the Model

```python
model.save(os.path.join('<insert path where you want to save your model>', 'imageclassifier.h5'))
new_model = load_model('imageclassifier.h5')
new_model.predict(np.expand_dims(resize / 255, 0))
```
- Saves the trained model to a specified path and then loads it back to ensure it was saved correctly. It makes a prediction with the reloaded model to verify its functionality.
---
The `app.py` creates a web interface for the fake fingerprint detection model using Gradio, which allows users to interact with the model by uploading images and receiving predictions. Here’s a detailed explanation of each part of the code:

### Imports

```python
import os
import tensorflow as tf
import keras
import cv2
import numpy as np
from keras.models import load_model
import gradio as gr
```
- The necessary libraries are imported: `os` for file operations, `tensorflow` and `keras` for deep learning, `cv2` for image processing, `numpy` for numerical operations, and `gradio` for creating the web interface.

### Load the Pre-trained Model

```python
model = load_model('<insert path of the saved model>')
```
- Loads the pre-trained model from the specified path. This model was previously trained and saved in the `FakeFingerprintDetection.py` script.

### Prediction Function

```python
def predict_image(image):
    resize = tf.image.resize(image, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat > 0.5:
        return 'Predicted as real'
    else:
        return 'Predicted as fake'
```
- This function takes an input image, resizes it to the required dimensions (256x256 pixels), normalizes the pixel values, and makes a prediction using the loaded model. It then returns whether the fingerprint is predicted as real or fake based on the model's output.

### Gradio Interface

```python
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Fake Finger Detection",
    description="Upload an image to check if it is a real or fake finger."
)
```
- This block sets up the Gradio interface:
  - `fn=predict_image`: Specifies that the `predict_image` function will be used to handle predictions.
  - `inputs=gr.Image(type="numpy")`: Defines the input type as an image, which will be converted to a NumPy array for processing.
  - `outputs="text"`: Defines the output type as text, which will display the prediction result.
  - `title="Fake Finger Detection"`: Sets the title of the interface.
  - `description="Upload an image to check if it is a real or fake finger."`: Provides a brief description for the interface.

### Launch the Interface

```python
interface.launch(share=True)
```
- Launches the Gradio interface, making it accessible to users. The `share=True` parameter allows the interface to be shared publicly via a unique URL, enabling others to use the model without needing to install the code locally.

This setup allows for a straightforward and interactive way to test the fake fingerprint detection model, making it accessible to a broader audience without requiring them to handle the underlying code directly.
---

This code represents a significant advancement in biometric security by employing deep learning techniques to effectively distinguish between real and fake fingerprints. By leveraging a convolutional neural network (CNN), it analyzes fingerprint images with high precision, thus mitigating the risks posed by spoofing attempts. The integration of this sophisticated model into a user-friendly Gradio interface further enhances accessibility and usability, allowing even non-technical users to verify fingerprint authenticity effortlessly. This evolution in spoof detection bolsters the security of fingerprint recognition systems, ensuring more reliable and robust protection against fraudulent access.
