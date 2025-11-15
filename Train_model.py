import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

audio = np.load("audio_features_aligned.npy")
image = np.load("image_features_aligned.npy")
bio = np.load("biometric_features_aligned.npy")
min_len = len(audio)
emotion_list = ['Happy', 'Sad', 'Angry', 'Fearful', 'Relaxed', 'Curious', 'Neutral']
n_classes = len(emotion_list)
labels = np.tile(np.arange(n_classes), int(np.ceil(min_len/n_classes)))[:min_len]
np.random.shuffle(labels)

le = LabelEncoder()
y = le.fit_transform(labels)
y_cat = to_categorical(y)

X_audio_tr, X_audio_te, X_img_tr, X_img_te, X_bio_tr, X_bio_te, y_tr, y_te = train_test_split(
    audio, image, bio, y_cat, test_size=0.2, random_state=42)

audio_input = Input(shape=(audio.shape[1], audio.shape[2]))
x_audio = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(audio_input)

image_input = Input(shape=(image.shape[1],))
x_image = BatchNormalization()(image_input)
x_image = Dense(128, activation='relu')(x_image)
x_image = BatchNormalization()(x_image)
x_image = Dropout(0.3)(x_image)

bio_input = Input(shape=(bio.shape[1],))
x_bio = BatchNormalization()(bio_input)
x_bio = Dense(32, activation='relu')(x_bio)
x_bio = BatchNormalization()(x_bio)
x_bio = Dropout(0.3)(x_bio)

merged = Concatenate()([x_audio, x_image, x_bio])
x = Dense(64, activation='relu')(merged)
x = Dropout(0.4)(x)
output = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=[audio_input, image_input, bio_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([X_audio_tr, X_img_tr, X_bio_tr], y_tr, epochs=30, batch_size=16, validation_split=0.3, verbose=0)

model.save("fusion_model.keras")
