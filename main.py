import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np
np.random.seed(10)

df = pd.read_csv("train.csv")

cols = ['Survived', 'Pclass', 'Name',
        'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
        'Embarked']
df = df[cols]

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

print('total:', len(df),
      'train:', len(train),
      'test:', len(test))

#資料標準化
def PreprocessData(raw_df):
    df = raw_df.drop(['Name'],axis=1)
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)

    fare_mean = df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(fare_mean)

    df['Sex'] = df['Sex'].map({'female':0 ,'male': 1}).astype(int)

    x_OneHot_df = pd.get_dummies(data=df, columns=["Embarked"])

    ndarry = x_OneHot_df.values
    Features = ndarry[:, 1:]
    Label = ndarry[:, 0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures, Label
# 測試資料
train_Features, train_Label = PreprocessData(train)
test_Features, test_Label = PreprocessData(test)


# Model
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x=train_Features,
                          y=train_Label,
                          validation_split=0.1,
                          epochs=30,
                          batch_size=30, verbose=2)
#model.save("TitanicModel")













