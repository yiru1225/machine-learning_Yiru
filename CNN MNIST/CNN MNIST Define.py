from tensorflow.python import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, save_model

# 1. Model configuration
batch_size = 600
img_width, img_height = 28, 28
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 1
optimizer = Adam()
validation_split = 0.2
verbosity = 1

# 2. Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# 3. Reshape data
input_train = input_train.reshape((input_train.shape[0], img_width, img_height, 1))
input_test = input_test.reshape((input_test.shape[0], img_width, img_height, 1))
input_shape = (img_width, img_height, 1)

# 4. Cast input to float32
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# 5. Normalize data
input_train = input_train / 255
input_test = input_test / 255

# 6. Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# 7. Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

# 8. Train the model
model.fit(input_train, target_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

# 9. Evaluate the model
scores = model.evaluate(input_train, target_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 10. Save the model（方法一）
model.save('model.h5')