from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os


os.chdir("/content/drive/MyDrive/Skenario 4")
#prepare data
data_dir =  '/content/drive/MyDrive/Skenario 4/TrainingPP'
test_dir = '/content/drive/MyDrive/Skenario 4/TestingPP'
target_size = (100,100)
classes = 26

train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    validation_split=0.2) #split data 20%
train_gen = train_datagen.flow_from_directory(
    data_dir, 
    target_size=target_size, 
    shuffle=True, 
    batch_size=32, 
    color_mode='rgb', 
    class_mode='categorical', 
    subset='training')
test_generator = ImageDataGenerator(
    rescale=1./255)
test_data_generator = test_generator.flow_from_directory(
    test_dir,
    target_size = (100,100),
    batch_size=32,
    shuffle=False)
val_gen = train_datagen.flow_from_directory(
    data_dir, 
    target_size=target_size, 
    batch_size=32 ,
    color_mode='rgb', 
    class_mode='categorical', 
    subset='validation' )

def make_model(ep):
  model = Sequential()
  #model 32 filer yang dihasilkan , ukuran kernel 3 , strides/pergeseran 1 , fungsi aktivasi relu , ukuran input 100x100,3(kernel)
  model.add(Conv2D(32 , kernel_size=3, strides=1, activation='relu', padding='same', input_shape=(100,100,3)))
  model.add(MaxPool2D(pool_size=(3,3),strides=2)) #pooling (pool size 3x3 ,pergeseran 2)

  model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same'))
  model.add(MaxPool2D(pool_size=(2,2),strides=2))

  model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same'))
  model.add(MaxPool2D(pool_size=(2,2),strides=2))
  model.add(Dropout(0.5))
  
  model.add(Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same'))
  model.add(MaxPool2D(pool_size=(2,2),strides=1))

  #FLATTEN
  model.add(Flatten())#input
  #Fullyconnected
  model.add(Dropout(0.5))
  model.add(Dense(512,activation='relu'))#input dense #512 neuron
  model.add(Dense(classes,activation='softmax'))

  model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
  plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  model.summary()

  #compile
  print("==== COMPILE ====")
  history = model.fit(train_gen,epochs=ep,validation_data=val_gen)
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()
  model_json=model.to_json()
  with open("model_json","w") as file:
      file.write(model_json)
  model.save_weights("model.h5")
  print("Model Saved Succesfully")

def evaluate_test():  
  json_file = open('model_json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  load_model = model_from_json(loaded_model_json)
  load_model.load_weights("model.h5")
  print("Model Loaded")

  prediksi = load_model.predict(test_data_generator)
  y_pred = np.argmax(prediksi,axis=1)
  class_labels = list(test_data_generator.class_indices.keys())   

  print("============== Tabel hasil Prediksi ===============")
  cm = confusion_matrix(test_data_generator.classes,y_pred)
  df_cm = pd.DataFrame(cm, index = class_labels,columns = class_labels)
  plt.figure(figsize=(15,15))
  sns.heatmap(df_cm, annot=True, annot_kws={"size": 9},cmap='viridis',fmt='g')

  print("===================== Hasil ====================")
  cr=classification_report(test_data_generator.classes,y_pred,target_names=class_labels,zero_division=1)
  print(cr)
  crv=classification_report(test_data_generator.classes,y_pred,target_names=class_labels,zero_division=1,output_dict=True)
  acc = crv['accuracy']
  acc_avg.append(acc)

acc_avg=[]
make_model(20)
evaluate_test()
print("Accuracy : ",acc_avg)
