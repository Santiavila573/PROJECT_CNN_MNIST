import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

# Evita que aparezcan mensajes innecesarios de TensorFlow mientras el programa corre.
# Esto mantiene la consola limpia y enfocada en los resultados importantes.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Cargar el conjunto de datos MNIST
# MNIST tiene 70,000 imágenes pequeñas (28x28 píxeles) de dígitos escritos a mano (0-9).
# Se divide en 60,000 imágenes para entrenar y 10,000 para probar el modelo.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocesar los datos
# Normalización: Convertimos los valores de los píxeles (0-255) a números entre 0 y 1.
# Esto ayuda al modelo a aprender más rápido y mejor.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Redimensionamos las imágenes para que tengan la forma (número de imágenes, 28, 28, 1).
# El "1" indica que las imágenes son en escala de grises (un solo canal de color).
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Convertimos las etiquetas (números del 0 al 9) a "one-hot encoding".
# Ejemplo: el número 2 se convierte en [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].
# Esto le dice al modelo que hay 10 clases distintas.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 3. Construir la red CNN (Convolutional Neural Network)
# Una CNN es ideal para imágenes porque detecta patrones como bordes y formas.
model = Sequential([  # "Sequential" significa que las capas se apilan en orden.
    # Primera capa convolucional: 32 filtros de 3x3 buscan patrones básicos en la imagen.
    # "relu" ayuda a detectar patrones no lineales.
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    # MaxPooling: Reduce el tamaño de la imagen tomando el valor máximo en áreas de 2x2.
    # Esto mantiene lo importante y baja el costo computacional.
    MaxPooling2D(pool_size=(2, 2)),
    # Segunda capa convolucional: 64 filtros de 3x3 buscan patrones más complejos.
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # Flatten: Convierte la imagen 2D en una lista 1D para las capas finales.
    Flatten(),
    # Capa densa: 128 neuronas conectan toda la información anterior.
    Dense(128, activation='relu'),
    # Dropout: Apaga el 50% de las neuronas al azar durante el entrenamiento.
    # Esto evita que el modelo memorice y lo obliga a generalizar.
    Dropout(0.5),
    # Capa de salida: 10 neuronas (una por dígito) con "softmax" para dar probabilidades.
    Dense(10, activation='softmax')
])

# Compilamos el modelo: Definimos cómo aprenderá.
# "adam" es un optimizador eficiente, "categorical_crossentropy" mide el error,
# y "accuracy" nos dice qué tan bien predice.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Entrenar la red
# EarlyStopping: Para el entrenamiento si no mejora tras 3 épocas.
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Entrenamos el modelo con los datos.
# epochs=10: Hará hasta 10 rondas de entrenamiento.
# batch_size=128: Procesa 128 imágenes a la vez.
# validation_split=0.2: Usa el 20% de los datos para validar.
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Evaluamos el modelo con los datos de prueba para ver su precisión final.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba: {test_accuracy:.4f}")

# Guardamos el modelo en un archivo para usarlo después sin reentrenar.
model.save('mnist_cnn_model.h5')

# 5. Predicciones
# Usamos el modelo para predecir los dígitos en las imágenes de prueba.
y_pred = model.predict(x_test)
# Convertimos las predicciones y etiquetas reales de one-hot a números enteros.
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
# Matriz de confusión: Muestra cuántas predicciones fueron correctas o incorrectas.
cm = confusion_matrix(y_test_classes, y_pred_classes)

# 6. Crear la interfaz gráfica
def mostrar_graficas():
    # Creamos una ventana con Tkinter para mostrar los resultados.
    ventana = tk.Tk()
    ventana.title("Resultados del Modelo MNIST")
    ventana.geometry("900x600")
    tab_control = ttk.Notebook(ventana)
    
    # Creamos tres pestañas para organizar los gráficos.
    tab1 = ttk.Frame(tab_control)
    tab2 = ttk.Frame(tab_control)
    tab3 = ttk.Frame(tab_control)
    tab_control.add(tab1, text='Curvas de Entrenamiento')
    tab_control.add(tab2, text='Matriz de Confusión')
    tab_control.add(tab3, text='Ejemplos de Clasificación')
    tab_control.pack(expand=1, fill='both')
    
    # Gráficas de entrenamiento: Muestran cómo mejoró el modelo.
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4))
    # Gráfico de pérdida: Qué tan mal estaba el modelo en cada época.
    ax1[0].plot(history.history['loss'], label='Training Loss')
    ax1[0].plot(history.history['val_loss'], label='Validation Loss')
    ax1[0].set_title('Loss vs. Epoch')
    ax1[0].set_xlabel('Epoch')
    ax1[0].set_ylabel('Loss')
    ax1[0].legend()
    # Gráfico de precisión: Qué tan bien predijo en cada época.
    ax1[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax1[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1[1].set_title('Accuracy vs. Epoch')
    ax1[1].set_xlabel('Epoch')
    ax1[1].set_ylabel('Accuracy')
    ax1[1].legend()
    
    # Guardamos la figura como imagen y la mostramos en la pestaña 1.
    canvas1 = tk.Canvas(tab1, width=600, height=400)
    canvas1.pack()
    plt.savefig("training_curves.png")
    img1 = tk.PhotoImage(file="training_curves.png")
    canvas1.create_image(10, 10, anchor=tk.NW, image=img1)
    
    # Matriz de confusión: Muestra errores y aciertos del modelo.
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2)
    ax2.set_title('Matriz de Confusión')
    ax2.set_xlabel('Predicción')
    ax2.set_ylabel('Etiqueta Real')
    
    # Guardamos la matriz como imagen y la mostramos en la pestaña 2.
    canvas2 = tk.Canvas(tab2, width=600, height=400)
    canvas2.pack()
    plt.savefig("confusion_matrix.png")
    img2 = tk.PhotoImage(file="confusion_matrix.png")
    canvas2.create_image(10, 10, anchor=tk.NW, image=img2)
    
    # Ejemplos de clasificación: Mostramos 5 imágenes con sus predicciones.
    fig3, ax3 = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        ax3[i].imshow(x_test[i].reshape(28, 28), cmap='gray')
        ax3[i].set_title(f"Pred: {y_pred_classes[i]}\nTrue: {y_test_classes[i]}")
        ax3[i].axis('off')
    
    # Guardamos los ejemplos como imagen y los mostramos en la pestaña 3.
    canvas3 = tk.Canvas(tab3, width=600, height=400)
    canvas3.pack()
    plt.savefig("sample_predictions.png")
    img3 = tk.PhotoImage(file="sample_predictions.png")
    canvas3.create_image(10, 10, anchor=tk.NW, image=img3)
    
    # Iniciamos la ventana para mostrar todo.
    ventana.mainloop()

# Llamamos a la función para abrir la interfaz gráfica.
mostrar_graficas()