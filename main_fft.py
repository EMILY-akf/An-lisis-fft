# ==========================================
# Análisis de señales con Transformada de Fourier
# Autor: (Tu nombre)
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Parámetros
# ===============================
fs = 1000  # Frecuencia de muestreo (Hz)
t = np.linspace(0, 1, fs)  # Tiempo

# ===============================
# 2. Definición de señales
# ===============================

# Señal senoidal
senal_seno = np.sin(2 * np.pi * 5 * t)

# Pulso rectangular
pulso = np.where((t > 0.4) & (t < 0.6), 1, 0)

# Función escalón
escalon = np.heaviside(t - 0.5, 1)

# ===============================
# 3. Función FFT
# ===============================
def calcular_fft(senal):
    fft = np.fft.fft(senal)
    frec = np.fft.fftfreq(len(senal), 1/fs)
    magnitud = np.abs(fft)
    fase = np.angle(fft)
    return frec, magnitud, fase

# ===============================
# 4. Calcular FFT de señales
# ===============================
f_seno, mag_seno, fase_seno = calcular_fft(senal_seno)
f_pulso, mag_pulso, fase_pulso = calcular_fft(pulso)
f_escalon, mag_escalon, fase_escalon = calcular_fft(escalon)

# ===============================
# 5. Función para graficar
# ===============================
def graficar(t, senal, f, mag, fase, titulo):
    plt.figure(figsize=(12, 8))

    # Dominio del tiempo
    plt.subplot(3,1,1)
    plt.plot(t, senal)
    plt.title("Señal en el tiempo - " + titulo)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

    # Magnitud
    plt.subplot(3,1,2)
    plt.plot(f, mag)
    plt.title("Magnitud de la FFT")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")

    # Fase
    plt.subplot(3,1,3)
    plt.plot(f, fase)
    plt.title("Fase de la FFT")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Fase")

    plt.tight_layout()
    plt.show()

# ===============================
# 6. Graficar resultados
# ===============================
graficar(t, senal_seno, f_seno, mag_seno, fase_seno, "Señal Senoidal")
graficar(t, pulso, f_pulso, mag_pulso, fase_pulso, "Pulso Rectangular")
graficar(t, escalon, f_escalon, mag_escalon, fase_escalon, "Función Escalón")

# ===============================
# 7. Propiedad de linealidad
# ===============================
senal_suma = senal_seno + pulso
f_suma, mag_suma, fase_suma = calcular_fft(senal_suma)

graficar(t, senal_suma, f_suma, mag_suma, fase_suma, "Suma de señales (Linealidad)")

print("Análisis completado correctamente.")
