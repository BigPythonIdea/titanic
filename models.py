# Model載入和運用
from tensorflow import keras
import main as m
# model = keras.models.load_model('TitanicModel')

import matplotlib.pyplot as plt
plt.plot(m.history.history['accuracy'])
plt.plot(m.history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(m.history.history['loss'])
plt.plot(m.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


