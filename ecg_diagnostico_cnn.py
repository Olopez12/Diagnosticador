import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dropout, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from imblearn.over_sampling import SMOTE
import datetime
import logging
import traceback

# Configuración de TensorFlow para evitar warnings innecesarios
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Función para descargar una base de datos si no está presente
def descargar_database(database_name, dl_dir):
    if not os.path.exists(dl_dir):
        try:
            print(f"Descargando la base de datos '{database_name}' en la carpeta '{dl_dir}'...")
            wfdb.dl_database(database_name, dl_dir=dl_dir)
            print("Descarga completada.")
        except Exception as e:
            print(f"Error al descargar la base de datos '{database_name}': {e}")
            traceback.print_exc()
    else:
        print(f"El directorio '{dl_dir}' ya existe. Asumiendo que el dataset ya está descargado.")

# Lista de bases de datos a descargar
databases_to_download = [
    {'name': 'mitdb', 'path': 'mitdb'},         # MIT-BIH Arrhythmia Database
    {'name': 'mitsvadb', 'path': 'mitsvadb'},   # MIT-BIH Supraventricular Arrhythmia Database
    {'name': 'mitltstdb', 'path': 'mitltstdb'}, # MIT-BIH Long-Term ECG Database
    {'name': 'mitnstdb', 'path': 'mitnstdb'},   # MIT-BIH Normal Sinus Rhythm Database
    # Añade más bases de datos según sea necesario
]

# Descargar todas las bases de datos
for db in databases_to_download:
    descargar_database(db['name'], db['path'])

# Definición de las bases de datos y sus registros
databases = [
    {
        'name': 'MIT-BIH Arrhythmia Database',
        'path': 'mitdb',
        'normal_records': [
            '100', '101', '103', '105', '112', '113', '114', '115', '116',
            '117', '121', '122', '123', '201', '202', '205', '208', '209',
            '215', '220', '223', '230', '231', '232', '234'
        ],
        'arrhythmia_records': [
            '104', '106', '107', '108', '109', '111', '118', '119', '124',
            '200', '203', '207', '210', '212', '213', '214', '219', '221',
            '222', '228', '233'
        ]
    },
    {
        'name': 'MIT-BIH Supraventricular Arrhythmia Database',
        'path': 'mitsvadb',
        'normal_records': [],  # Generalmente, todos los registros contienen arritmias
        'arrhythmia_records': [
            '800', '801', '802', '803', '804', '805', '806', '807', '808',
            '809', '810', '811', '812', '820', '821', '822', '823', '824',
            '825', '826', '827', '828', '829', '840', '841', '842', '843',
            '844', '845', '846', '847', '848', '849', '850', '851', '852',
            '853', '854', '855', '856', '857', '858', '859', '860', '861',
            '862', '863', '864', '865', '866', '867', '868', '869', '870',
            '871', '872', '873', '874', '875', '876', '877', '878', '879',
            '880', '881', '882', '883', '884', '885', '886', '887', '888',
            '889', '890', '891', '892', '893', '894'
        ]
    },
    {
        'name': 'MIT-BIH Long-Term ECG Database',
        'path': 'mitltstdb',
        'normal_records': [], 
        'arrhythmia_records': [
            '14046', '14134', '14149', '14157', '14172', '14184', '15814'
        ]
    },
    {
        'name': 'MIT-BIH Normal Sinus Rhythm Database',
        'path': 'mitnstdb',
        'normal_records': [
            '16265', '16272', '16273', '16420', '16483', '16539', '16773', '16786', '16795',
            '17052', '17453', '18177', '18184', '19088', '19090', '19093', '19140', '19830'
        ],
        'arrhythmia_records': []  # Generalmente, no contienen arritmias
    },
    
    # Añade más bases de datos según sea necesario
]

# Función para diseñar y aplicar un filtro pasa-bajas
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Función para extraer características de un registro
def extract_features(record_name, label, dl_dir):
    try:
        # Lectura del registro
        record = wfdb.rdrecord(os.path.join(dl_dir, record_name), channels=[0])
        signal = record.p_signal.flatten()
        fs = record.fs  # Frecuencia de muestreo

        # Filtrado pasa-bajas
        cutoff = 45  # Frecuencia de corte
        order = 5
        filtered_signal = butter_lowpass_filter(signal, cutoff, fs, order)

        # Normalización
        normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

        # Detección de picos R
        peaks_R, _ = find_peaks(normalized_signal, distance=fs*0.2, height=0.5)

        # Características de intervalos RR
        rr_intervals = np.diff(peaks_R) / fs
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
        std_rr = np.std(rr_intervals) if len(rr_intervals) > 0 else 0
        heart_rate = 60 / mean_rr if mean_rr > 0 else 0  # Frecuencia cardíaca en bpm

        # Análisis de frecuencia
        freqs, psd = welch(normalized_signal, fs, nperseg=1024)
        total_power = np.sum(psd)
        dominant_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0

        # Características de las ondas P y T (simplificado)
        # Detección de ondas P y T como picos más pequeños entre las ondas R
        peaks_P, _ = find_peaks(-normalized_signal, distance=fs*0.2, height=0.2)
        peaks_T, _ = find_peaks(normalized_signal, distance=fs*0.2, height=0.2, prominence=0.1)

        # Características de la onda P
        if len(peaks_P) > 1:
            p_wave_duration = np.mean(np.diff(peaks_P) / fs)
            p_wave_amplitude = np.mean(normalized_signal[peaks_P])
        else:
            p_wave_duration = 0
            p_wave_amplitude = 0

        # Características de la onda T
        if len(peaks_T) > 1:
            t_wave_duration = np.mean(np.diff(peaks_T) / fs)
            t_wave_amplitude = np.mean(normalized_signal[peaks_T])
        else:
            t_wave_duration = 0
            t_wave_amplitude = 0

        # Compilación de características
        features = [
            mean_rr,
            std_rr,
            heart_rate,
            total_power,
            dominant_freq,
            p_wave_duration,
            p_wave_amplitude,
            t_wave_duration,
            t_wave_amplitude
        ]
        return features, label, normalized_signal
    except Exception as e:
        print(f"Error al procesar el registro {record_name} en {dl_dir}: {e}")
        return None, None, None

# Función personalizada para padding de señales
def pad_signals(signals, max_length):
    """
    Rellena las señales ECG para que todas tengan la misma longitud.

    Args:
        signals (list of np.array): Lista de señales ECG.
        max_length (int): Longitud máxima deseada.

    Returns:
        np.array: Array de señales rellenadas con forma (n_samples, max_length, 1).
    """
    # Truncar y rellenar las señales
    signals_padded = pad_sequences(
        signals,
        maxlen=max_length,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    
    # Añadir una dimensión para el canal
    signals_padded = signals_padded.reshape(signals_padded.shape[0], signals_padded.shape[1], 1)
    return signals_padded

# Definir la longitud máxima para las señales ECG
MAX_LENGTH = 5000  # Puedes ajustar este valor según tus necesidades

# Construcción del dataset
data = []
labels = []
signals = []

# Iterar sobre las bases de datos y registros definidos
for db in databases:
    db_name = db['name']
    db_path = db['path']
    normal_records = db['normal_records']
    arrhythmia_records = db['arrhythmia_records']
    
    print(f"\nExtrayendo características de registros normales de {db_name}...")
    for record_name in normal_records:
        features, label, signal = extract_features(record_name, 0, dl_dir=db_path)  # 0 para normal
        if features is not None:
            data.append(features)
            labels.append(label)
            signals.append(signal)

    print(f"Extrayendo características de registros con arritmia de {db_name}...")
    for record_name in arrhythmia_records:
        features, label, signal = extract_features(record_name, 1, dl_dir=db_path)  # 1 para arritmia
        if features is not None:
            data.append(features)
            labels.append(label)
            signals.append(signal)

# Verificar si se extrajeron datos
if not data:
    raise ValueError("No se extrajeron características de ningún registro. Revisa la descarga de los datasets.")

# Conversión a arrays de NumPy
data = np.array(data)
labels = np.array(labels)
signals = np.array(signals, dtype=object)

print(f"\nTotal de muestras extraídas: {len(data)}")

# Normalización de las características
scaler = StandardScaler()
data = scaler.fit_transform(data)

# División en conjunto de entrenamiento y prueba para modelos clásicos
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.3, random_state=42, stratify=labels
)

# Preparación de los datos para la CNN
# Padding de las señales para que tengan la misma longitud
signals_padded = pad_signals(signals, MAX_LENGTH)

# División en conjunto de entrenamiento y prueba para la CNN
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    signals_padded, labels, test_size=0.3, random_state=42, stratify=labels
)

# Balanceo del conjunto de datos para la CNN utilizando SMOTE
print("\nBalanceando el conjunto de datos para la CNN usando SMOTE...")
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(
    X_train_cnn.reshape(X_train_cnn.shape[0], -1), y_train_cnn
)

# Restaurar la forma original de las señales
X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], MAX_LENGTH, 1)

print(f"  Conjunto de entrenamiento balanceado: {X_train_resampled.shape[0]} muestras")
print(f"  Conjunto de prueba: {X_test_cnn.shape[0]} muestras")

# Construcción del modelo CNN mejorado
model = Sequential()
model.add(Input(shape=(MAX_LENGTH, 1)))  # Usar Input layer para evitar warnings
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilación del modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Integración con TensorBoard y Early Stopping
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento del modelo CNN
print("\nEntrenando modelo CNN...")
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=50,  # Incrementado a 50 para permitir mayor aprendizaje
    batch_size=16,
    validation_data=(X_test_cnn, y_test_cnn),
    callbacks=[tensorboard_callback, early_stopping]
)

# Evaluación del modelo CNN
loss, accuracy = model.evaluate(X_test_cnn, y_test_cnn)
print(f"\nExactitud del modelo CNN: {accuracy:.4f}")

# Predicciones CNN
y_pred_cnn = model.predict(X_test_cnn)
y_pred_cnn = (y_pred_cnn > 0.5).astype(int).flatten()

# Reporte de clasificación CNN
print("\nReporte de clasificación para CNN:")
print(classification_report(y_test_cnn, y_pred_cnn, target_names=['Normal', 'Arritmia']))

# Matriz de confusión CNN
cm_cnn = confusion_matrix(y_test_cnn, y_pred_cnn)
plt.figure(figsize=(6,5))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Normal', 'Arritmia'], yticklabels=['Normal', 'Arritmia'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - CNN')
plt.show()

# Visualización de las curvas de entrenamiento CNN
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend(loc='upper left')
plt.show()

# Entrenamiento y evaluación con modelos clásicos

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

print("\nEntrenando modelo Decision Tree...")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predicción y evaluación Decision Tree
y_pred = clf.predict(X_test)
print("Exactitud del modelo Decision Tree:", accuracy_score(y_test, y_pred))
print("Reporte de clasificación para Decision Tree:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Arritmia']))

# Matriz de confusión Decision Tree
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Arritmia'], yticklabels=['Normal', 'Arritmia'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - Decision Tree')
plt.show()

# Random Forest
from sklearn.ensemble import RandomForestClassifier

print("\nEntrenando modelo Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("Exactitud del modelo Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Reporte de clasificación para Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Arritmia']))

# Matriz de confusión Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Arritmia'], yticklabels=['Normal', 'Arritmia'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - Random Forest')
plt.show()

# Validación cruzada con Random Forest
from sklearn.model_selection import cross_val_score

print("\nRealizando validación cruzada con Random Forest...")
scores = cross_val_score(
    rf_clf,
    data, labels,
    cv=5,
    scoring='accuracy'
)
print("Exactitud promedio con Validación Cruzada:", scores.mean())