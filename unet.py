import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Carica immagine e maschera
image_path = "D:/Desktop/Adas_test/convert/ampera/images/amz_00820.png"
mask_path = "test.jpg"

# Carica immagini (ridimensiona per il modello)
image = cv2.imread(image_path) / 255.0  # Normalizza tra 0 e 1
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0  # Normalizza maschera

# Ridimensionamento per compatibilità (se necessario)
image = cv2.resize(image, (256, 256))
mask = cv2.resize(mask, (256, 256))

# Espandi dimensioni per batch (aggiunge dimensione batch)
image = np.expand_dims(image, axis=0)
mask = np.expand_dims(mask, axis=-1)
mask = np.expand_dims(mask, axis=0)

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Contrazione
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Espansione
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = concatenate([conv2, up4], axis=-1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([conv1, up5], axis=-1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit del modello
history = model.fit(image, mask, epochs=10, batch_size=1)

# Previsione sulla stessa immagine di input
pred_mask = model.predict(image)

# Rimozione della dimensione batch e conversione in formato immagine
pred_mask = np.squeeze(pred_mask)  # Rimuove dimensione 1 del batch
pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Threshold e scala a 0-255

# Salvataggio della maschera predetta
output_path = "/mnt/data/predicted_mask.jpg"
cv2.imwrite(output_path, pred_mask)

print(f"Maschera predetta salvata in: {output_path}")

def iou_metric(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).astype(np.uint8)
    intersection = np.sum((y_true * y_pred))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union

# Calcolo IoU sulla maschera di test
iou_score = iou_metric(mask[0], pred_mask)
print(f"IoU Score: {iou_score:.4f}")


# Mostra immagine originale e maschera predetta
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Immagine Originale")
plt.imshow(image[0])

plt.subplot(1, 2, 2)
plt.title("Maschera Predetta")
plt.imshow(pred_mask, cmap='gray')

plt.show()
