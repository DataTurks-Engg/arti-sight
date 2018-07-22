from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
import os
import pickle
from keras.models import model_from_json
import matplotlib.pyplot as plt

image_width, image_height= 256, 256

nb_train_samples= 11000
nb_validation_sample=2000
batch_size = 8

model = applications.VGG19(weights= "imagenet", include_top=False, input_shape=(image_height, image_width,3))

for layer in model.layers[:5]:
	layer.trainable = False

x=model.output
x=Flatten()(x)
x=Dense(1024, activation="relu")(x)
x=Dropout(0.5)(x)
x=Dense(384, activation="relu")(x)
x=Dropout(0.5)(x)
x=Dense(96, activation="relu")(x)
x=Dropout(0.5)(x)
predictions = Dense(30, activation="softmax")(x)


model_final =Model(input=model.input, output=predictions)
#model_final = load_model("weights_VGG.h5")
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.nadam(lr=0.00001), metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode="nearest",
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   rotation_range=30)

test_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

training_set = train_datagen.flow_from_directory('./HE_Chal',
                                                 target_size = (256, 256),
                                                 batch_size = 8,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./Validation',
                                            target_size = (256, 256),
                                            batch_size = 8,
                                            class_mode = 'categorical')
                                            
model_final.fit_generator(training_set,                         steps_per_epoch = 1000,epochs = 80,                         validation_data = test_set,validation_steps=1000)
model_json=model_final.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model_final.save_weights("weights_VGG.h5")
model_final.save("model_27.h5")
#model_final.predict(test_set, batch_size=batch_size)
'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights_VGG.h5",by_name=True)
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

#print(loaded_model.summary())
loaded_model.fit_generator(training_set,                         steps_per_epoch = 1000,epochs = 100,                         validation_data = test_set,validation_steps=1000)
#score = loaded_model.evaluate(training_set,test_set , verbose=0)
'''
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
