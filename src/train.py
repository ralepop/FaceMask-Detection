import os
import matplotlib.pyplot as plt
from data_loader import extract_data, get_class_distribution, get_data_generators
from model_builder import build_cnn, compile_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(BASE_DIR, 'dataset.zip')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset/data')
DATASET_EXTRACT_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models/mask_detector_v1.keras')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
EPOCHS = 15
BATCH_SIZE = 32

def main():
    extract_data(ZIP_PATH, DATASET_EXTRACT_DIR)

    # histogram
    get_class_distribution(DATASET_DIR, REPORTS_DIR)

    # generatori
    train_gen, val_gen = get_data_generators(DATASET_DIR, batch_size = BATCH_SIZE)

    # kreiranje modela
    model = build_cnn(input_shape = (128, 128, 3))
    model = compile_model(model)

    # ispis
    model.summary()

    # trening
    print("Pocetak treninga...")
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    print(f"Model sacuvan na: {MODEL_SAVE_PATH}")

    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs = EPOCHS
    )

    model.save(MODEL_SAVE_PATH)
    print(f"Model uspesno istreniran i sacuvan na: {MODEL_SAVE_PATH}")

    # grafik performansi
    plt.figure(figsize = (12, 4))

    # tacnost
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label = 'Trening Tacnost')
    plt.plot(history.history['val_accuracy'], label = 'Validaciona Tacnost')
    plt.title('Tacnost tokom epoha')
    plt.legend()

    # gubitak (loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Trening Gubitak')
    plt.plot(history.history['val_loss'], label = 'Validacioni Gubitak')
    plt.title('Gubitak tokom epoha')
    plt.legend()

    plt.savefig(os.path.join(REPORTS_DIR, 'performance_graphs.png'))
    plt.show()

if __name__ == "__main__":
    main()