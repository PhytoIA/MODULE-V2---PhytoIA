import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.models import Sequential
print(tf.config.list_physical_devices())

training_set = tf.keras.utils.image_dataset_from_directory(
   'train',
   labels="inferred",
   label_mode="categorical", #Classes (38)
   class_names=None,
   color_mode="rgb",
   batch_size=32,
   image_size=(224, 224),
   shuffle=True,
   seed=None,
   validation_split=None,
   subset=None,
   interpolation="bilinear",
   follow_links=False,
   crop_to_aspect_ratio=False,
)

validation_set = tf.keras.utils.image_dataset_from_directory(
   'valid',
   labels="inferred",
   label_mode="categorical",
   class_names=None,
   color_mode="rgb",
   batch_size=32,
   image_size=(224, 224),
   shuffle=True,
   seed=None,
   validation_split=None,
   subset=None,
   interpolation="bilinear",
   follow_links=False,
   crop_to_aspect_ratio=False,
)

model = tf.keras.models.load_model("trained_model_efficient.keras")
base_model = model.layers[1]

model.summary()

# for i, layer in enumerate(model.layers): #Layers of the model to find EfficientNet index
#     print(i, layer.name, type(layer))

# for i, layer in enumerate(base_model.layers): #Layers of the base model (EfficientNet)
#     print(i, layer.name)

# Tout geler
base_model.trainable = False

# Dégeler les 30 dernières couches
for layer in base_model.layers[-20:]:
   if not isinstance(layer, tf.keras.layers.BatchNormalization):
      layer.trainable = True # Si il ne s'agit pas d'une couche de BatchNormalization, entraîner la couche.
   else:
      layer.trainable = False # Aussi non, non (pour éviter la baisse d'accuracy).
   print(layer.name, layer.trainable)

print("Couches entraînables :", sum([layer.trainable for layer in base_model.layers]))

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])

# Model Training
training_history = model.fit(x=training_set,validation_data=validation_set,epochs=5)

# Model Evaluation on Training set
train_loss, train_acc = model.evaluate(training_set)

print(train_loss, train_acc)

#Model on Validation set
val_loss, val_acc = model.evaluate(validation_set)

print(val_loss, val_acc)

#Saving model
model.save("trained_model_efficient_20layers_finetuned.keras") # Change name if retraining
training_history.history

#Recording History in json
import json
with open("trained_model_efficient_20layers_finetuned.json", "w") as f:
    json.dump(training_history.history, f)

training_history.history['val_accuracy']

epochs = [i for i in range (1,6)]
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy Result")
plt.title("Visualization of Accuracy Result - Fine-Tuning")
plt.legend()
plt.show()

# Some other metrics for model evaluation
class_name = validation_set.class_names

test_set = tf.keras.utils.image_dataset_from_directory(
   'valid',
   labels="inferred",
   label_mode="categorical", #Classes (38)
   class_names=None,
   color_mode="rgb",
   batch_size=32,
   image_size=(224, 224),
   shuffle=False,
   seed=None,
   validation_split=None,
   subset=None,
   interpolation="bilinear",
   follow_links=False,
   crop_to_aspect_ratio=False,
)

y_pred = model.predict(test_set)
y_pred,y_pred.shape

predicted_categories = tf.argmax(y_pred,axis=1)
true_categories = tf.concat([y for x,y in test_set], axis=0)

Y_true = tf.argmax(true_categories,axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(Y_true, predicted_categories, target_names=class_name))

cm = confusion_matrix(Y_true, predicted_categories)
cm.shape

# Confusion Matrix Visualization
plt.figure(figsize=(40,40))
sns.heatmap(cm, annot=True, annot_kws={'size': 10})
plt.xlabel("Predicted Class", fontsize=20)
plt.ylabel("Actual Class", fontsize=20)
plt.title("Plant Disease Prediction Confusion Matrix", fontsize=25)
plt.show()

# FINE TUNING --------------------------------------------------------
# def unfreeze_model(model):
#     # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
#     for layer in model.layers[-20:]:
#         if not isinstance(layer, layers.BatchNormalization):
#             layer.trainable = True

#     optimizer = keras.optimizers.Adam(learning_rate=1e-5)
#     model.compile(
#         optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
#     )

# unfreeze_model(model)

# epochs = 4  # @param {type: "slider", min:4, max:10}
# hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)
# plot_hist(hist) --------------------------------------------------------
# Meilleur modèle: 
# checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     "best_efficientnet_finetuned.keras",
#     monitor="val_accuracy",
#     save_best_only=True,
#     mode="max"
# )
# model.fit(
#     training_set,
#     validation_data=validation_set,
#     epochs=5,
#     callbacks=[checkpoint]
# )