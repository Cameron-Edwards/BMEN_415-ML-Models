

import numpy as np
import os 
import cv2
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow import keras
from keras import layers


def ImportData(Data_Directory, IMG_SIZE):
#Importing the Training Data

    DataDir = Data_Directory
    CATEGORIES = ["NORMAL", "PNEUMONIA"]
    training_data = []

    for category in CATEGORIES:
        path = os.path.join(Data_Directory, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

    np.random.shuffle(training_data)

    X = []
    Y = []

    for inputs, labels in training_data:
        X.append(inputs)
        Y.append(labels)

    return np.array(X), np.array(Y)

x_train, y_train = ImportData("C:/Users/camer/Documents/BMEN 415/Final Project/ImageDataset/chest_xray/train",200)

# creating a plot
pixel_plot = plt.figure()
  
# plotting a plot
  
# customizing plot
# plt.title("Chest X-rays")
# pixel_plot = plt.imshow(x_train[0], cmap='gray')
# plt.show()
# print(y_train.shape)



x_val, y_val_a = ImportData("C:/Users/camer/Documents/BMEN 415/Final Project/ImageDataset/chest_xray/val",200)
x_test, y_test_a = ImportData("C:/Users/camer/Documents/BMEN 415/Final Project/ImageDataset/chest_xray/test",200)

y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test_a, 2)
y_val = np_utils.to_categorical(y_val_a, 2)

print(y_test)

print(x_train.shape)

input_shape = (200, 200, 1)
# the data, split between train and test sets

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_val = x_val.astype("float32") / 255

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 5


opt = keras.optimizers.Adam(lr=0.00001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test))

plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()


plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.show()


model.save('./CNN_model')



def confusion_matrix(true, pred):
    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result

test_pred = model.predict(x_test)
test_pred_t = np.empty_like(test_pred)
test_pred_t = test_pred_t[:,1]

for i in range(test_pred.shape[0]):
    test_pred_t[i] = np.argmax(test_pred[i])
test_pred_t = test_pred_t.astype(int)
print("test_pred_t", test_pred_t)
con_matrix = confusion_matrix(y_test_a, test_pred_t)

print(con_matrix)

# weather_pred = model.predict(img_list_2)
# weather_pred_t = np.empty_like(weather_pred)
# weather_pred_t = weather_pred_t[:,1]
# for i in range(weather_pred.shape[0]):
#     weather_pred_t[i] = np.argmax(weather_pred[i])
#
# weather_pred_t = weather_pred_t.astype(int)
# labels_2 = labels_2.astype(int)
# confusion_mx2 = compute_confusion_matrix(labels_2, weather_pred_t)
# print(confusion_mx2)