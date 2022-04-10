import csv
import numpy
import cv2
from matplotlib import pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Flatten
import tensorflow.python.keras as keras
import tensorflow.python.keras.applications
from tensorflow.keras.models import Model
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from numba import cuda
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K, optimizers
from sklearn.model_selection import StratifiedKFold, KFold
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing import image

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

scheme = 'Geetest/'
root = 'Data_set/'

with open("Data_set/字符集.txt", 'r', encoding='utf-8') as f:
    chars = f.read()
    chars_set = list(chars)
char_to_num = {}
i = 0
for c in chars_set:
    char_to_num[c] = i
    i += 1
num_to_char = {v: k for k, v in char_to_num.items()}


# %% 读取数据
def get_path(picture_dir, label_dir):
    picture_path = []
    label_path = []
    for name in os.listdir(picture_dir):
        picture_path.append(picture_dir + '/' + name)
        label = label_dir + '/' + name[:-3] + 'txt'
        label_path.append(label)
    return picture_path, label_path


class data_generator:
    def __init__(self, all_picture_path, labels, image_size, batch_size=None):
        self.index = 0
        self.all_picture_path = all_picture_path
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size

    def get_mini_batch(self):
        while True:
            batch_images = []
            batch_labels = []
            for i in range(self.batch_size):
                if self.index == len(self.all_picture_path):
                    self.index = 0
                bgr_image = cv2.imread(self.all_picture_path[self.index])
                if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
                    bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                rgb_image = cv2.resize(rgb_image, (self.image_size[0], self.image_size[1]))
                img = np.array(rgb_image).astype(np.float32)
                img /= 127.5
                img -= 1.
                batch_images.append(img)
                label_path = self.labels[self.index]
                f = open(label_path, 'r', encoding='utf-8')
                label = f.read()
                batch_labels.append(label)
                f.close()
                self.index += 1
            l = pd.DataFrame(batch_labels)
            y = l[0].map(char_to_num)
            batch_labels = to_categorical(y, 3755)
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield batch_images, batch_labels


shape = (80, 80, 3)  # Inception
classes = 3755
batch_size = 4
lr = 0.00003
check_pointer = ModelCheckpoint(filepath='models/Geetest/model_vgg/best_model.hdf5',
                                monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback_list = [check_pointer]

# %%
picture_path, labels = get_path(root + scheme + 'X_train', root + scheme + 'y_train')
train_data = data_generator(picture_path, labels, shape, batch_size)
steps_per_epoch = len(picture_path) // batch_size
picture_path, labels = get_path(root + scheme + 'X_val', root + scheme + 'y_val')
val_data = data_generator(picture_path, labels, shape, batch_size)
val_step = len(picture_path) // batch_size

# %% 调用resnet进行训练

base_model = tensorflow.keras.applications.VGG16(include_top=True, weights=None, classes=classes,
                                                 input_shape=shape,
                                                 )
# inp = Input(shape)
# x = base_model(inp)
# x = GlobalAveragePooling2D()(x)
# x = Flatten()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(classes, activation='softmax')(x)

model = base_model

for layer in base_model.layers:
    layer.trainable = True

opt = optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x=train_data.get_mini_batch(), steps_per_epoch=steps_per_epoch, epochs=50,
                    validation_data=val_data.get_mini_batch(),
                    validation_steps=val_step, callbacks=callback_list)

model.save('models/Geetest/model_vgg/final_model.hdf5')

loss_trend_graph_path = "models/Geetest/model_vgg/_loss.jpg"
acc_trend_graph_path = "models/Geetest/model_vgg/_acc.jpg"
fig = plt.figure(1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(acc_trend_graph_path)
plt.close(1)
# summarize history for loss
fig = plt.figure(2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(loss_trend_graph_path)
plt.close(2)
