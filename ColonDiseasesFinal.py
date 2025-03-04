#Gerekli kütüphaneleri import ediyoruz.

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D


train_dir = 'ColonDiseases/train'
test_dir = 'ColonDiseases/test'


batch_size = 64 #Her iterasyonda fotoğraflar 64lük paketler halinde gidiyor
img_height, img_width = 224, 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True,
    seed=123
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(224, 224),
    layers.Rescaling(1./255)
])



class_names = train_ds.class_names
print(f"Sınıf İsimleri: {class_names}")

#Dataları çoğaltma işlemi yapıyoruz ve plota basıyoruz.


data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomContrast(0.2),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8")) 
        plt.axis("off")




train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# EfficientnetB5 modelini kullanıyoruz.
pretrained_model = tf.keras.applications.EfficientNetB5(
    include_top=False,          
    weights='imagenet',        # ImageNet veri kümesiyle önceden eğitildi
    input_shape=(224, 224, 3)  
)


pretrained_model.trainable = False

#CheckPoint oluşturma
checkpoint_path = "ColonDiseasesWeights.weights.h5"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor="val_accuracy",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)


# EfficentNetb5 in üstüne sınıflandırma kaktmanımızı ekliyoruz. 

inputs = pretrained_model.input
x = GlobalAveragePooling2D()(pretrained_model.output)  
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  
outputs = Dense(4, activation='softmax')(x)  # 4 Sınıf için son katmanımız.


model = Model(inputs=inputs, outputs=outputs)

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Modelin özeti
#model.summary() # model yapısını basar 

# Modeli eğitmeye başlıyoruz 10 epoch ile
epochs = 10

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=test_ds
)



# Eğitim ve doğrulama  sonuçlarını plota basıyoruz

plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Kaybı: {test_loss}")
print(f"Test Doğruluğu: {test_accuracy}")

# Test verisinden data alıyoruz

test_images, test_labels = next(iter(test_ds))
predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=1)


predicted_labels = [class_names[k] for k in predictions]
true_labels = [class_names[np.argmax(label)] for label in test_labels]


#rastgele 64 görüntü seçme
random_indices = np.random.choice(len(test_images), 64, replace=False)
fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(15, 15))

for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[random_indices[i]].numpy().astype("uint8"))
    true_label = true_labels[random_indices[i]]
    pred_label = predicted_labels[random_indices[i]]
    
    # doğruysa yeşil değilse kırmızı işaretliyoruz
    color = "green" if true_label == pred_label else "red"
    ax.set_title(f"Gerçek: {true_label}\nTahmin: {pred_label}", color=color)
    ax.axis('off')

plt.tight_layout()
plt.show()





"""

maks 4 sayfa

aynı font

en sona foto cv





"""







