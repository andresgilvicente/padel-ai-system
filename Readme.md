<div align="center">
  
  <a href="https://github.com/jorgecarnicero/ProyectoFinalComputerVision">
    <img src="assets/banner_readme.png" alt="Padel AI System Banner" width="100%">
  </a>

  <h1>🎾 Padel AI System</h1>
  <p><strong>Visión por Ordenador aplicada a la Seguridad y el Análisis Deportivo</strong></p>

  <p>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+">
    </a>
    <a href="https://streamlit.io/">
      <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App">
    </a>
    <a href="https://opencv.org/">
      <img src="https://img.shields.io/badge/OpenCV-CV2-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
    </a>
    <a href="https://ultralytics.com/yolov8">
      <img src="https://img.shields.io/badge/YOLOv8-Pose-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLOv8 Pose">
    </a>
  </p>
</div>

<hr />


## 📖 Introducción

**Padel AI System** es un proyecto integral de Visión por Ordenador desarrollado para la asignatura *Visión por Ordenador I* en la **Universidad Pontificia Comillas (ICAI)**

Este sistema implementa una solución de dos módulo utilizando una única cámara de un dispositivo móvil:
1.  **Sistema de Seguridad:** Un módulo de control de acceso basado en el reconocimiento de patrones geométricos que actúa como una contraseña visual.
2.  **Sistema de Tracking (Juez de Silla IA):** Un motor de análisis deportivo autónomo capaz de rastrear una pelota de pádel, detectar jugadores mediante estimación de pose y validar saques (Válido/Falta) en tiempo real.

## 📂 Estructura del Repositorio

El proyecto está organizado en directorios modulares para garantizar la escalabilidad y el orden:

```text
├── 📂 assets/                   # Imágenes y contenido visual, por ejemplo para el banner del ReadMe.md
├── 📂 calibration_process/      # Scripts e imágenes para la calibración intrínseca de la cámara
├── 📂 complete_padel_system/    # Aplicación unificada integrando Seguridad + Tracker
├── 📂 security_system/          # Módulo independiente de reconocimiento de patrones geométricos
├── 📂 tracking_system/          # Módulo independiente de seguimiento de bola y lógica de arbitraje
├── 📄 .gitignore                # Configuración de Git
├── 📄 ReadMe.md                 # Documentación del proyecto
└── 📄 documentation.pdf         # Informe final y documentación del proyecto
└── 📄 requirements.txt          # Dependencias y librerías necesarias
```

## 🛠️ Tecnologías y Metodología

El sistema se basa en un enfoque híbrido que combina **Visión por Ordenador Clásica** y algoritmos más avanzados como los que ofrece la librería de **YOLO**:

* **Core Framework:** Python 3, OpenCV (cv2).
* **Deep Learning:** YOLOv8-Pose (Ultralytics) para la extracción de puntos clave del jugador (cintura/pies).


* **Técnicas Clásicas:**
    * Segmentación de color HSV y sustracción de fondo MOG2 para detección de la pelota.
    * Flujo Óptico (Lucas-Kanade) para el suavizado de trayectorias.
    * Filtros de Kalman para la predicción de estado y manejo de oclusiones.
    * Aproximación geométrica (Douglas-Peucker) sobre el *Convex Hull* para el módulo de seguridad.
    * Uso de operaciones morfológicos, thresholding, binarización, etc.


* **Interfaz:** Streamlit para el dashboard web y visualización en tiempo real.
* **Optimización:** Multihilo (*Threading*) para la captura de vídeo y soporte opcional de TensorRT para la inferencia.

## 🚀 Instalación y Configuración

### Requisitos Previos

Asegúrese de tener instalado **Python 3.9** o superior.

### 1. Clonar el Repositorio

```bash
git clone https://github.com/jorgecarnicero/ProyectoFinalComputerVision.git

```

### 2. Crear Entorno Virtual (Recomendado)

```bash
python -m venv venv # En Windows

.\venv\Scripts\activate # En Mac/Linux

source venv/bin/activate

```

### 3. Instalar Dependencias

Todas las librerías necesarias se encuentran listadas en `requirements.txt`.

```bash
pip install -r requirements.txt

```

## 🖥️ Ejecución

Puede ejecutar los módulos de forma independiente o como un sistema completo.

### Opción A: Sistema Completo (Seguridad + Tracker)

Ejecuta el flujo completo. Deberá superar el control de seguridad (mostrar 4 formas geométricas) para desbloquear el tracker.

```bash
streamlit run complete_padel_system/complete_padel_system_app.py

```

### Opción B: Sistema de Tracking (Solo Árbitro)

Lanza directamente el Juez de Silla IA para análisis o depuración.

```bash
streamlit run tracking_system/tracking_system_app.py

```

### Opción C: Sistema de Seguridad

Prueba la lógica de reconocimiento de patrones geométricos de forma aislada.

```bash
streamlit run security_system/security_system_app.py

```

## 📊 Descripción de Funcionalidades

🔒 Módulo de Seguridad 

* **Detección de Formas:** Identifica Líneas, Triángulos, Cuadrados, Rectángulos, Círculos y Pentágonos mediante análisis de contornos y *convex hulls*.
* **Decodificador de Secuencia:** Desbloquea el sistema únicamente cuando se detecta una secuencia específica de 4 formas geométricas predefinidas.

🎾 Módulo de Tracking (Juez IA) 

* **Seguimiento de Bola:** Detección híbrida usando Color/Movimiento + Predicción por Filtro de Kalman.
* **Validación de Saque:**
  * Detecta el impacto del saque basado en picos de aceleración.
  * Compara la altura de la pelota vs. la altura de la cintura del jugador (Keypoints YOLO).
  * Clasifica el saque como **VALID (Válido)** o **FAULT (Falta)**.


* **Detección de Bote:** Analiza la trayectoria en el eje Y para detectar cambios de dirección (rebotes) en la pista.
* **Generación de Evidencia:** Guarda automáticamente fotogramas "Foto Finish" de cada saque analizado.

## 👥 Autores

* **Jorge Carnicero Príncipe**
* **Andrés Gil Vicente** 


