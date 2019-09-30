import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

df = pd.read_csv("housepricedata.csv")

dataset = df.values

X = dataset[:,0:10]
Y = dataset[:,10]

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_train,X_val_and_test,Y_train,Y_val_and_test = train_test_split(X_scale,Y,test_size=0.3)
X_val,X_test,Y_val,Y_test = train_test_split(X_val_and_test,Y_val_and_test,test_size=0.5)

model = Sequential([Dense(32,activation='relu',input_shape=(10,)),
            Dense(32,activation='relu'),
            Dense(1,activation='sigmoid'),])

model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])


hist = model.fit(X_train,Y_train,batch_size=32,epochs=90,validation_data=(X_val,Y_val))
model.evaluate(X_test,Y_test)[1]

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc="upper right")
plt.show()


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()