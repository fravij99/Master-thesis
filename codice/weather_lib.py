import requests
import pandas as pd
import imageio
import moviepy.editor as mp
from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import base64
import requests
from requests.auth import HTTPBasicAuth
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

def dataTest():
    API_KEY = "1177877c41514a008a1c6614300b34bb"

    data_test = {
        "lat": 45.509,
        "lon": 9.187,
        "start_date": "2024-01-08",
        "end_date": "2024-01-09",
        "tz": "local",
        "key": API_KEY,
    }
    return data_test

def dataTrue():
    API_KEY = "1177877c41514a008a1c6614300b34bb"

    data = {
        "lat": 45.509,
        "lon": 9.187,
        "start_date": "2020-08-08",
        "end_date": "2024-08-09",
        "tz": "local",
        "key": API_KEY,
    }
    return data

def make_request(data, url):
    try:
        response = requests.get(url, params=data)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data_json = response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        print(data_json)


    print(f'The response of the url is:{response}') 
    # Trova la lunghezza massima tra le liste
    max_length = max(len(value) for value in data_json.values() if isinstance(value, list))
    # Filtra le colonne con lunghezze diverse
    filtered_data_json = {key: value for key, value in data_json.items() if isinstance(value, list) and len(value) == max_length}
    #df=pd.DataFrame(filtered_data_json)
    df = pd.DataFrame.from_dict(filtered_data_json)
    data_df = pd.json_normalize(df['data'])

    # Combina i due DataFrame
    result_df = pd.concat([df, data_df], axis=1)

    #   Rimuovi la colonna 'data' originale se necessario
    result_df = result_df.drop(columns=['data'])

    result_df.reset_index(drop=True, inplace=True)

    result_df.to_csv("datiMeteo.csv")
    return response.status_code, result_df


def extract_frames(video_path, output_folder):
    # Carica il video utilizzando moviepy
    video = mp.VideoFileClip(video_path)

    # Estrai i frame
    def extract_frame(t):
        return video.get_frame(t)

    frames = []
    with ThreadPoolExecutor() as executor:
        for frame in tqdm(executor.map(extract_frame, range(0, int(video.duration), 1)), total=int(video.duration), desc="Estrazione frame"):
            frames.append(frame)

    # Salva i frame nella cartella specificata
    def save_frame(i_frame_frame):
        i, frame = i_frame_frame
        file_path = os.path.join(output_folder, f"frame_{i + 1}.png")
        imageio.imwrite(file_path, frame)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(save_frame, enumerate(frames)), total=len(frames), desc="Salvataggio frames"))

    print(f"Frames estratti e salvati in: {output_folder}")



def save_frames(folder_path):

    # Lista per memorizzare i frames
    frames = []

    # Leggi i frame dalla cartella e salvali nella lista
    for filename in tqdm(os.listdir(folder_path)):  # Utilizza tqdm per creare una barra di avanzamento
        if filename.endswith(".png"):  # Assicurati che i file siano immagini PNG
            file_path = os.path.join(folder_path, filename)
            frame = imageio.imread(file_path)
            frames.append(frame)
    
    frames=frames[:-1]

    return frames

def show_frames(path_csv, frames):
    df = pd.read_csv(path_csv)
    df.reset_index(drop=True, inplace=True)

    # Aggiungi una colonna al DataFrame per le immagini
    df['Images'] = frames

    # Possiamo accedere all'immagine associata a una riga specifica del DataFrame
    indice_riga = 209
    immagine_associata = df.at[indice_riga, 'Images']

    # Visualizza l'immagine associata
    #plt.imshow(immagine_associata, cmap='gray')  # Assumendo immagini in scala di grigi
    #plt.title(f"Umidità relativa: {df.at[indice_riga, 'rh']} %")
    #plt.show()
    return df

# Funzione per ridurre la risoluzione di un'immagine
def resize_image(image, new_width, new_height):
    new_dimensions = (new_width, new_height)
    resized_image = cv2.resize(image, new_dimensions)
    
    return resized_image

def resize_dataset(df, new_width, new_height):
    df['Images'] = [resize_image(image, new_width, new_height) for image in df['Images']]
    df['Images'] = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for frame in df['Images']]

def binarization_and_split(variable, images):
    bin_variable = (variable > np.mean(variable)).astype(int)


    # Split dei dati in train e test
    X_train, X_test, y_train, y_test = train_test_split(images, bin_variable, test_size=0.2, random_state=42)

    # Converti le immagini in numpy arrays e aggiungi una dimensione per il canale (scala di grigi)
    X_train = np.expand_dims(np.stack(X_train.values), axis=-1)
    X_test = np.expand_dims(np.stack(X_test.values), axis=-1)
    y_train = np.expand_dims(np.stack(y_train.values), axis=-1)
    y_test = np.expand_dims(np.stack(y_test.values), axis=-1)

    return X_train, X_test, y_train, y_test

def prec_and_split(prec, images):

    precip = (prec != 0 ).astype(int)

    # Split dei dati in train e test
    X_train, X_test, y_train, y_test = train_test_split(images, precip, test_size=0.2, random_state=42)

    # Converti le immagini in numpy arrays e aggiungi una dimensione per il canale (scala di grigi)
    X_train = np.expand_dims(np.stack(X_train.values), axis=-1)
    X_test = np.expand_dims(np.stack(X_test.values), axis=-1)
    y_train = np.expand_dims(np.stack(y_train.values), axis=-1)
    y_test = np.expand_dims(np.stack(y_test.values), axis=-1)

    return X_train, X_test, y_train, y_test


def create_cnn(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    layers.Dropout(0.2)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    layers.Dropout(0.2)
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    layers.Dropout(0.2)
    model.add(layers.Dense(1))  # Ultimo layer con un singolo neurone per la regressione
    return model


def train_cnn(X_train, X_test, y_train, y_test, epochs):
    sns.set_style('darkgrid')

    model=create_cnn(X_train.shape[1:])
    # Compila il modello
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Addestra il modello
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    plot_history(history)
    plt.savefig('history.png')
    plt.show()

    # Valuta il modello
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Accuracy sul set di test: {test_acc}')