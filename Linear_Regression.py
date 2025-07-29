from sklearn.model_selection import train_test_split
import pandas as pd
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras import layers

# df=keras.datasets.boston_housing

(x_train,y_train),(x_test,y_test)=keras.datasets.california_housing.load_data()

# normalization
std_scalar=StandardScaler()
x_train_scaled=std_scalar.fit_transform(x_train)
x_test_scaled=std_scalar.transform(x_test)

feature=8

model=Sequential([
    keras.Input(shape=(feature,)),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='linear')
])

model.compile(optimizer='adam',loss='mean_squared_error')

print(model.summary())

fitting=model.fit(x_train_scaled,y_train,
                  epochs=100,
                  batch_size=32,
                  validation_split=0.2,
                  verbose=1
)