from tensorflow.keras import layers, models

def build_cnn(input_shape = (128, 128, 3)):
    model = models.Sequential([
        # 1. konvolucioni blok - izdvajanje osnovnih karakteristika
        layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape),
        layers.MaxPooling2D((2,2)),

        # 2. konvolucioni blok - slozenije karakterisike
        layers.Conv2D(64, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2,2)),

        # 3. konvolucioni blok
        layers.Conv2D(128, (3,3), activation = 'relu'),
        layers.MaxPooling2D((2,2)),

        # smanjujemo broj parametara i rizik od preobucavanja
        layers.GlobalAveragePooling2D(),

        # dropout
        layers.Dense(128, activation = 'relu'),
        layers.Dropout(0.5),    # nasumicno gasimo 50% neurona tokom treninga

        # izlazni sloj: 1 neuron sa sigmoid aktivacionom funkcijom (binarna klasifikacija)
        layers.Dense(1, activation = 'sigmoid')
    ])

    return model

def compile_model(model):
    # optimizator: adam (RMSProp + Momentum)
    # loss: binary crossentropy
    model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )
    return model