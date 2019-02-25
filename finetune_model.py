from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from loading_models import base_model


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


class_list = [
    "Advan",
    "Alcatel",
    "Asus",
    "Blackberry",
    "Brandcode",
    "Evercross",
    "Honor",
    "Huawei",
    "Icherry",
    "Infinix",
    "Iphone",
    "Lenovo",
    "Maxtron",
    "Mito",
    "Motorola",
    "Nokia",
    "Oppo",
    "Others Mobile & Tablet",
    "Realme",
    "Samsung",
    "Sharp",
    "Smartfren",
    "Sony",
    "SPC",
    "Strawberry",
    "Vivo",
    "Xiaomi"
]
FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))
