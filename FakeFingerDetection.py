# IMPORT
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import load_model
import cv2

# Load data
data_dir = 'train'
data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(256, 256), batch_size=32)

# Visualize some images
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# Data preprocessing
data = data.map(lambda x, y: (x / 255, y))

# Calculate dataset sizes
data_size = tf.data.experimental.cardinality(data).numpy()
train_size = int(data_size * 0.7)
val_size = int(data_size * 0.15)
test_size = data_size - train_size - val_size

# Adjust sizes to ensure non-zero splits
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

# Split data
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Ensure the sizes are correct
train_count = tf.data.experimental.cardinality(train).numpy()
val_count = tf.data.experimental.cardinality(val).numpy()
test_count = tf.data.experimental.cardinality(test).numpy()
#print(f'Train size: {train_count}')
#print(f'Validation size: {val_count}')
#print(f'Test size: {test_count}')


# Model building
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

# Train model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Plot performance
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

#plt.show()

# Evaluate
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}')
print(f'Recall: {re.result().numpy()}')
print(f'Accuracy: {acc.result().numpy()}')

# Test the model
img = cv2.imread('<insert test image file path>')
resize = tf.image.resize(img, (256, 256))

yhat = model.predict(np.expand_dims(resize / 255, 0))

if yhat > 0.5:
    print('Predicted as real')
else:
    print('Predicted as fake')

# Save the model
model.save(os.path.join('<insert path where you want to save your model>', 'imageclassifier.h5'))
new_model = load_model('imageclassifier.h5')
new_model.predict(np.expand_dims(resize / 255, 0))