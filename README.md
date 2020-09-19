# 鐵達尼號生存預測

>本檔案分成Main 和 model
>>使用者可以直接取用TitanicModel訓練好的Model直接使用

## 準確度參考


| loss     | accuracy | val_loss | val_accuracy |
| -------- | -------- | -------- | --------     |
| 0.4601   | 0.7975   | 0.3949   | 0.8451       |


## 使用套件

```
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
```
## 資料來源Kaggle Titantic
[https://www.kaggle.com/c/titanic](https://)



## 資料標準化

```
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
```
> 引用至:TensorFlow+Keras深度學習人工智慧實務應用


## Model 訓練
```
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x=train_Features,
                          y=train_Label,
                          validation_split=0.1,
                          epochs=30,
                          batch_size=30, verbose=2)
```
> 引用至:TensorFlow+Keras深度學習人工智慧實務應用



