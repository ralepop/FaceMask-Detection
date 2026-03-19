# ucitavanje, analiza i augmentacija

import zipfile
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def extract_data(zip_path, extract_to = 'dataset/'):
    if not os.path.exists(extract_to) or not os.listdir(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Uspesno otpakovano u {extract_to}")
    else:
        print("Dataset vec postoji.")

def get_class_distribution(data_path, reports_dir=None):
    # prolazimo kroz foldere i brojimo slike za izvestaj
    classes = ['with_mask', 'without_mask']

    # filtriramo samo foldere
    counts = [len([f for f in os.listdir(os.path.join(data_path, c)) if not f.startswith('.')]) for c in classes]

    # histogram
    plt.figure(figsize=(8, 5))
    plt.bar(classes, counts, color = ['#4CAF50', '#F44336'])
    plt.title('Raspodela uzoraka po klasama')
    plt.ylabel('Broj slika')

    if reports_dir:
        save_path = os.path.join(reports_dir, 'histogram_klasa.png')
        plt.savefig(save_path)
        print(f"Histogram sacuvan na: {save_path}")

    plt.show()
    return counts

def get_data_generators(data_path, target_size = (128, 128), batch_size = 32):
    # pretprocesiranje i augmentaciju (zastita od preobucavanja)
    datagen = ImageDataGenerator(
        rescale = 1./255,           # normalizacija [0, 255] -> [0, 1]
        rotation_range = 20,        # nasumicna rotacija
        width_shift_range = 0.1,    # horiz. pomeranje
        height_shift_range = 0.1,   # vert. pomeranje
        horizontal_flip = True,     # nasumicno okretanje slike
        validation_split = 0.2      # automatsko odvajanje 20% za test/validaciju
    )

    train_gen = datagen.flow_from_directory(
        data_path,
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary',      # jer imamo dve klase
        subset = 'training',
        shuffle = True
    )

    val_gen = datagen.flow_from_directory(
        data_path,
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary',
        subset = 'validation',
        shuffle = False             # ne mesamo validaciju
    )

    return train_gen, val_gen