#%%
# 1. Import packages
from keras.preprocessing.image import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import os, datetime
import numpy as np
import random
import cv2


#%%
# 2. Data loading
data = r"C:\Users\artsc\Desktop\TensorFlow\Assessment3\Concrete Crack Images for Classification"
neg_class = r"C:\Users\artsc\Desktop\TensorFlow\Assessment3\Concrete Crack Images for Classification\Negative"
pos_class = r"C:\Users\artsc\Desktop\TensorFlow\Assessment3\Concrete Crack Images for Classification\Positive"


#%%
# 3. Display the number in each image class
neg = len(os.listdir(neg_class))
pos = len(os.listdir(pos_class))
print(f"The number of Negative images: {neg}")
print(f"The number of Positive images: {pos}")

plt.bar(["Negative","Positive"],[neg,pos])
plt.title("Class Balancing")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


#%%
# 4. Display some images as examples
neg_list = os.listdir(neg_class)
random.shuffle(neg_list)
neg_list = neg_list[:10]

pos_list = os.listdir(pos_class)
random.shuffle(pos_list)
pos_list = pos_list[:10]

count = 1
plt.figure(figsize=(20,10))
for item in neg_list:
    plt.subplot(2,5,count)
    img = cv2.imread(neg_class + "/" + item)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    count = count + 1
    plt.title("Negative Class - No cracks")

count = 1
plt.figure(figsize=(20,10))
for item in pos_list:
    plt.subplot(2,5,count)
    img = cv2.imread(pos_class + "/" + item)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    count = count + 1
    plt.title("Positive Class - Cracks")



#%%
# 6. Model creation

model = keras.Sequential()

# (A) Input layer
model.add(layers.InputLayer(input_shape=img.shape))

# (B) Feature extractor
model.add(keras.layers.Conv2D(8,(3,3),activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(3,3))
model.add(keras.layers.Conv2D(16,(3,3),activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(3,3))
model.add(keras.layers.Conv2D(32,(3,3),activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(3,3))

# (C) Classifier
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(2,activation='softmax'))

model.summary()



#%%
#7. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



train_gen = image_dataset_from_directory(directory=data,
                                         batch_size=64,
                                         image_size=(227, 227))
test_gen = image_dataset_from_directory(directory=data,
                                        batch_size=1,
                                        image_size=(227, 227))
rescale = Rescaling(scale=1.0/255)
train_gen = train_gen.map(lambda image,label:(rescale(image),label))
test_gen  = test_gen.map(lambda image,label:(rescale(image),label))



#%%
# 8. Setup TensorBoard callback
log_path = os.path.join('log_dir', 'concrete_crack', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_path)
es = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True, verbose=1)


#%%
# 9. Model training
EPOCHS = 5
history = model.fit(train_gen, validation_data=test_gen,
          epochs = EPOCHS, callbacks=[tb,es])


#%%
# 10. Model deployment
y_pred = np.argmax(model.predict(test_gen), axis=1)


#%%
# 11. Model save

model.save('model.h5')
# %%
