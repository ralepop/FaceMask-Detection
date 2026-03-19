import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn
from data_loader import get_data_generators
from train import MODEL_SAVE_PATH, DATASET_DIR, REPORTS_DIR


def evaluate_model(model_path, dataset_dir):
    # ucitavanje modela i podataka
    model = load_model(model_path)
    _, val_gen = get_data_generators(dataset_dir)

    # predikcije
    print("Radim predikcije na validacionom skupu...")

    y_true = val_gen.classes
    y_pred_prob = model.predict(val_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # matrica konfuzije
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize = (8, 6))
    seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['With Mask', 'Without Mask'],
                    yticklabels=['With Mask', 'Without Mask'])
    plt.xlabel('Predvidjeno')
    plt.ylabel('Stvarno')
    plt.title('Matrica konfuzije')
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
    plt.show()

    # classification report (tacnost, preciznost)
    print("\nKlasifikacioni izvestaj:")
    print(classification_report(y_true, y_pred, target_names=['With Mask', 'Without Mask']))

    visualize_errors(val_gen, y_true, y_pred)

def visualize_errors(val_gen, y_true, y_pred):
    # uzima se jedan batch slika
    images, _ = next(val_gen)
    # mapiranje labela za prikaz
    label_map ={0: 'With Mask', 1: 'Without Mask'}

    plt.figure(figsize=(12, 10))
    for i in range(9):      # prikazuje se prvih 9 slika iz batch-a
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])

        color = 'green' if y_true[i] == y_pred[i] else 'red'
        title_text = f"True: {label_map[y_true[i]]}\nPred: {label_map[y_pred[i]]}"

        plt.title(title_text, color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'sample_predictions.png'))
    plt.show()

if __name__ == "__main__":
    evaluate_model(MODEL_SAVE_PATH, DATASET_DIR)