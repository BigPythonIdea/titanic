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



