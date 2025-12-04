# Sistema Híbrido de Detección Emocional: LBPH + Análisis Geométrico de Landmarks Faciales

## 📋 Requisitos Previos

### Software Necesario
- Python 3.8 o superior instalado en su sistema
- Webcam funcional (integrada o externa)
- Sistema Operativo: Windows, Linux o macOS

### Instalación de Dependencias

Abra una terminal o símbolo del sistema e instale las bibliotecas requeridas ejecutando el comando de instalación de opencv-contrib-python y numpy.

### Archivo Opcional (Recomendado para Mayor Precisión)

Descargue el modelo de detección de landmarks faciales llamado lbfmodel.yaml desde el repositorio GSOC2017 en GitHub. Este archivo pesa 68.7 MB y debe colocarse en el mismo directorio que el script principal.

**Nota:** El sistema funciona sin este archivo, pero con menor precisión en el análisis geométrico facial.

---

## 🚀 Ejecución del Sistema

### Iniciar el Programa

Ejecute el archivo sistemas_emociones.py. El sistema mostrará un menú principal con tres opciones:
- Opción 1: Agregar/Entrenar persona
- Opción 2: Modo detección de emociones
- Opción 3: Salir

---

## 🎓 Modo 1: Entrenamiento (Primera Vez)

### Proceso de Captura de Emociones

**Paso 1:** Seleccione la opción 1 del menú principal.

**Paso 2:** Ingrese su nombre cuando el sistema lo solicite. Este nombre identificará su modelo personalizado.

**Paso 3:** Presione ENTER para iniciar el proceso de captura.

**Paso 4:** Comenzará la captura de video en tiempo real. El sistema mostrará su rostro con detección de puntos faciales.

**Paso 5:** Para capturar una foto, presione la tecla ESPACIO. El sistema le pedirá que identifique la emoción que está expresando:
- Número 0 para Enojado 😠
- Número 1 para Feliz 😊
- Número 2 para Neutral 😐
- Número 3 para Triste 😢
- Número 4 para Sorprendido 😮

**Paso 6:** Repita el proceso hasta capturar mínimo 3 fotos de cada emoción (se recomienda 5 fotos por emoción para mejor precisión).

**Paso 7:** Una vez capturadas todas las emociones suficientemente, presione la tecla C para completar el entrenamiento.

**Paso 8:** El sistema procesará las imágenes, extraerá características y entrenará el modelo. Este proceso toma solo unos segundos.

### Controles del Modo Entrenamiento
- **ESPACIO:** Capturar fotografía
- **C:** Completar y guardar entrenamiento
- **ESC:** Cancelar proceso

### Consejos para un Buen Entrenamiento
- Mantenga buena iluminación frontal en su rostro
- Colóquese entre 50-100 cm de distancia de la cámara
- Asegúrese de que solo aparezca un rostro en pantalla
- Exagere ligeramente las expresiones para mayor claridad
- Capture cada emoción con diferentes intensidades

### Resultado
El sistema creará una carpeta con su nombre dentro del directorio emociones_data y guardará el modelo entrenado y todas las fotografías capturadas organizadas por emoción.

---

## 👁️ Modo 2: Detección en Tiempo Real

### Proceso de Detección

**Paso 1:** Seleccione la opción 2 del menú principal.

**Paso 2:** Ingrese el nombre exacto de la persona que fue previamente entrenada.

**Paso 3:** El sistema cargará el modelo entrenado. Si no existe, mostrará un error indicando que debe entrenar primero.

**Paso 4:** Presione ENTER para iniciar la detección en tiempo real.

**Paso 5:** El sistema comenzará a analizar su rostro y mostrará:
- Su video en tiempo real
- Un recuadro alrededor de su rostro con color específico para cada emoción
- El emoji y nombre de la emoción detectada
- El porcentaje de confianza de la predicción
- Los 68 puntos faciales marcados en verde
- Estadísticas del sistema (FPS, cantidad de rostros)

### Controles del Modo Detección
- **Q:** Salir del modo detección y volver al menú
- **ESPACIO:** Pausar o reanudar la detección
- **F:** Activar o desactivar la visualización de puntos faciales

### Interpretación de Resultados
- **Recuadro Verde:** Emoción Feliz detectada
- **Recuadro Rojo:** Emoción Enojado detectada
- **Recuadro Gris:** Emoción Neutral detectada
- **Recuadro Naranja:** Emoción Triste detectada
- **Recuadro Amarillo:** Emoción Sorprendido detectada

El porcentaje mostrado indica la confianza del sistema en su predicción. Valores mayores al 70% indican alta confianza.

---

## 📁 Organización de Archivos

El sistema genera automáticamente una estructura de carpetas:
- Una carpeta principal llamada emociones_data
- Dentro, una subcarpeta con el nombre de cada persona entrenada
- Dentro de cada persona, carpetas individuales para cada emoción con las fotos capturadas
- Un archivo de base de datos que contiene el modelo entrenado

---

## 🔧 Solución de Problemas Comunes

### Error de Cámara No Detectada
Verifique que su webcam esté conectada correctamente y funcionando. Pruebe abrir otra aplicación que use la cámara para confirmar que funciona. Revise los permisos de acceso a la cámara en la configuración de su sistema operativo.

### Error de Bibliotecas No Instaladas
Si el sistema indica que falta opencv-contrib-python, deberá desinstalar cualquier versión de opencv-python regular e instalar específicamente la versión contrib que incluye los módulos adicionales necesarios.

### Advertencia de Archivo lbfmodel.yaml Faltante
El sistema continuará funcionando pero con menor precisión. Para obtener los mejores resultados, descargue este archivo y colóquelo en la misma carpeta del programa.

### No Se Detecta el Rostro Durante Captura
Asegúrese de tener iluminación adecuada. La luz debe venir de frente, no de atrás (evite estar contra una ventana o luz trasera). Ajuste su distancia a la cámara. Verifique que solo haya una persona en el encuadre.

### Baja Precisión en Detección
Si el sistema no detecta correctamente sus emociones, puede mejorar el modelo capturando más fotografías. Simplemente ejecute nuevamente la opción 1 con el mismo nombre y agregue más ejemplos de cada emoción.

---

## 📊 Información del Sistema

### Rendimiento
- Procesa video a 30 cuadros por segundo en hardware estándar
- Precisión superior al 80% con modelos personalizados bien entrenados
- Tiempo de entrenamiento menor a 10 minutos
- Detección instantánea sin retraso perceptible

### Características Técnicas
- Analiza 5 estados emocionales diferentes
- Extrae 11 características geométricas del rostro
- Identifica 68 puntos faciales de referencia
- Combina análisis de textura facial y geometría

### Privacidad y Seguridad
- Todos los datos se almacenan localmente en su computadora
- No se envía información a internet
- El modelo es personal y no funciona con otras personas
- Puede eliminar sus datos borrando la carpeta con su nombre

---

## 🎯 Mejores Prácticas

### Durante el Entrenamiento
- Capture al menos 5 fotos de cada emoción para mejor precisión
- Varíe ligeramente la intensidad de cada expresión
- Mantenga la misma iluminación y posición de cámara que usará después
- Sea consistente con su apariencia (si usa lentes, úselos en todas las fotos)

### Durante el Uso Regular
- Use el sistema en las mismas condiciones de iluminación del entrenamiento
- Mantenga la cámara en la misma posición utilizada durante la captura
- Haga expresiones claras y sostenidas para mejor detección
- Si cambia significativamente su apariencia, considere reentrenar el modelo

### Para Múltiples Usuarios
- Cada persona debe entrenar su propio modelo con su nombre único
- No intente usar el modelo de otra persona, la precisión será muy baja
- Los modelos son completamente independientes entre usuarios

---

## 🆘 Ayuda Adicional

Si experimenta problemas técnicos:
- Revise que Python esté correctamente instalado verificando la versión
- Confirme que las bibliotecas estén instaladas correctamente
- Verifique los permisos de su sistema operativo para acceso a cámara
- Lea los mensajes de error en la consola, estos proporcionan información específica del problema
- Asegúrese de tener suficiente espacio en disco para almacenar las imágenes capturadas

---

## 📝 Notas Importantes

- El modelo entrenado es específico para cada individuo y no se puede compartir entre usuarios
- La calidad de la detección depende directamente de la calidad del entrenamiento inicial
- Puede mejorar su modelo en cualquier momento agregando más fotografías de entrenamiento
- El sistema funciona mejor con expresiones faciales claras y deliberadas
- Los datos permanecen completamente privados en su dispositivo local
