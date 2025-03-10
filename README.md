#  Implementación de una CNN para reconocimiento de dígitos con MNIST
![convolution](https://github.com/user-attachments/assets/a0248f11-3505-41ce-97c3-947813bdd14f)

## Descripción
Este proyecto implementa una Red Neuronal Convolucional (CNN) para clasificar imágenes de dígitos escritos a mano del conjunto de datos MNIST. Utiliza Python y TensorFlow para cargar, preprocesar, entrenar y evaluar el modelo. Además, incluye una interfaz gráfica para visualizar los resultados del entrenamiento, la matriz de confusión y ejemplos de clasificación. Es una herramienta útil para aprender sobre redes neuronales y procesamiento de imágenes.

<img src="https://github.com/user-attachments/assets/f632530d-ce3a-48ea-9f9f-52d0d8feffd3" width="600" alt="fclayer">

## Características
- Carga y preprocesamiento del conjunto de datos MNIST.
- Construcción y entrenamiento de una CNN con capas convolucionales y densas.
- Evaluación de la precisión del modelo en el conjunto de prueba.
- Visualización de curvas de entrenamiento, matriz de confusión y ejemplos de clasificación mediante una interfaz gráfica.

<img src="https://github.com/user-attachments/assets/2894c268-fdba-40d8-bcba-6708f5e15629" width="500" alt="softmax">

## Requisitos
Para ejecutar este proyecto, necesitarás lo siguiente:

- **Python**: Versión 3.8 o superior
- **Bibliotecas**:
  - TensorFlow 2.x
  - NumPy
  - Matplotlib
  - Seaborn
  - Tkinter

<img src="https://github.com/user-attachments/assets/34953f33-3de7-443c-ad52-baab47dd444f" width="300" alt="image">

## Instalación
Sigue estos pasos para configurar el proyecto en tu máquina local:

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Santiavila573/PROJECT_CNN_MNIST.git
   ## Navega al directorio del proyecto:

## Navega al directorio del proyecto:
```bash
cd reconocimiento-digitos-mnist
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate
pip install -r requirements.txt
```
## Uso
Una vez configurado el entorno, puedes ejecutar el proyecto con los siguientes pasos:

1. Asegúrate de que el entorno virtual esté activado (si lo creaste).
2. Ejecuta el script principal:
   ```bash
   python main.py
### Estructura del Proyecto

![image](https://github.com/user-attachments/assets/bf3e9091-8056-4372-8ae8-85f9adeba919)
  
### Resultados
- Precisión del Modelo: El modelo alcanza una precisión aproximada de 98-99% en el conjunto de prueba.
- Curvas de Entrenamiento: Muestran la pérdida y precisión a lo largo de las épocas.
- Matriz de Confusión: Visualiza los aciertos y errores del modelo.
- Ejemplos de Clasificación: Exhiben imágenes de dígitos con sus predicciones y etiquetas reales.

<img src="https://github.com/user-attachments/assets/b5451195-c605-49ee-b348-3d5098c3d4b5" width="600" alt="pooling-1">

### Licencia
- Este proyecto está licenciado bajo la Licencia MIT.
### Contacto
- Para preguntas, sugerencias o comentarios, por favor contactar a: santiago.avila@iti.edu.ec
