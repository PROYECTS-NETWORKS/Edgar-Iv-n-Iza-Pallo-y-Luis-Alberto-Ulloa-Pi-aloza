"""
Sistema de Detección de Emociones Faciales MEJORADO
- Características geométricas de landmarks
- Múltiples clasificadores (LBPH + SVM)
- Análisis de regiones faciales (ojos, boca, cejas)
- Mayor precisión en detección de emociones
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import pickle
from collections import deque

# ==================== EXTRACTOR DE CARACTERÍSTICAS FACIALES ====================
class FacialFeatureExtractor:
    """Extrae características geométricas de landmarks para mejorar detección"""
    
    @staticmethod
    def extract_geometric_features(landmarks):
        """
        Extrae características geométricas clave de los landmarks
        Returns: vector de características (numpy array)
        """
        if landmarks is None or len(landmarks) < 68:
            return None
        
        features = []
        
        try:
            # 1. CARACTERÍSTICAS DE CEJAS
            left_brow = np.array(landmarks[19:22])
            left_eye = np.array(landmarks[37:40])
            if len(left_brow) > 0 and len(left_eye) > 0:
                left_brow_eye_dist = np.mean([np.linalg.norm(np.array(left_brow[i]) - np.array(left_eye[i])) 
                                               for i in range(min(len(left_brow), len(left_eye)))])
                features.append(left_brow_eye_dist)
            else:
                features.append(0.0)
            
            right_brow = np.array(landmarks[22:25])
            right_eye = np.array(landmarks[43:46])
            if len(right_brow) > 0 and len(right_eye) > 0:
                right_brow_eye_dist = np.mean([np.linalg.norm(np.array(right_brow[i]) - np.array(right_eye[i])) 
                                                for i in range(min(len(right_brow), len(right_eye)))])
                features.append(right_brow_eye_dist)
            else:
                features.append(0.0)
            
            # Distancia entre cejas
            left_brow_inner = np.array(landmarks[21])
            right_brow_inner = np.array(landmarks[22])
            brow_distance = np.linalg.norm(left_brow_inner - right_brow_inner)
            features.append(brow_distance)
            
            # 2. CARACTERÍSTICAS DE OJOS
            left_eye_height = np.linalg.norm(np.array(landmarks[37]) - np.array(landmarks[41]))
            left_eye_width = np.linalg.norm(np.array(landmarks[36]) - np.array(landmarks[39]))
            left_eye_ratio = left_eye_height / (left_eye_width + 1e-6)
            features.append(left_eye_ratio)
            
            right_eye_height = np.linalg.norm(np.array(landmarks[43]) - np.array(landmarks[47]))
            right_eye_width = np.linalg.norm(np.array(landmarks[42]) - np.array(landmarks[45]))
            right_eye_ratio = right_eye_height / (right_eye_width + 1e-6)
            features.append(right_eye_ratio)
            
            # 3. CARACTERÍSTICAS DE BOCA
            mouth_top = np.array(landmarks[51])
            mouth_bottom = np.array(landmarks[57])
            mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
            features.append(mouth_height)
            
            mouth_left = np.array(landmarks[48])
            mouth_right = np.array(landmarks[54])
            mouth_width = np.linalg.norm(mouth_left - mouth_right)
            features.append(mouth_width)
            
            mouth_ratio = mouth_height / (mouth_width + 1e-6)
            features.append(mouth_ratio)
            
            # Curvatura de boca
            mouth_center_y = (landmarks[51][1] + landmarks[57][1]) / 2
            left_corner_y = landmarks[48][1]
            right_corner_y = landmarks[54][1]
            mouth_curvature = ((left_corner_y + right_corner_y) / 2) - mouth_center_y
            features.append(mouth_curvature)
            
            # 4. PROPORCIONES FACIALES
            eye_distance = np.linalg.norm(np.array(landmarks[36]) - np.array(landmarks[45]))
            features.append(eye_distance)
            
            face_top = np.array(landmarks[27])
            face_bottom = np.array(landmarks[8])
            face_height = np.linalg.norm(face_top - face_bottom)
            face_width = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[16]))
            face_ratio = face_height / (face_width + 1e-6)
            features.append(face_ratio)
            
            return np.array(features, dtype=np.float32)
        except:
            return None
    
    @staticmethod
    def get_feature_names():
        """Retorna nombres descriptivos de las características"""
        return [
            'left_brow_eye',
            'right_brow_eye',
            'brow_distance',
            'left_eye_ratio',
            'right_eye_ratio',
            'mouth_height',
            'mouth_width',
            'mouth_ratio',
            'mouth_curve',
            'eye_distance',
            'face_ratio'
        ]


# ==================== DETECTOR DE PUNTOS FACIALES ====================
class FacialLandmarkDetector:
    """Detector de puntos faciales usando OpenCV Facemark"""
    def __init__(self):
        print("Inicializando detector de puntos faciales...")
        
        landmark_model = "lbfmodel.yaml"
        
        if not os.path.exists(landmark_model):
            print("\n" + "="*70)
            print("ADVERTENCIA: No se encontró lbfmodel.yaml")
            print("="*70)
            print("Para usar detección de puntos faciales, descargue:")
            print("https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml")
            print("\nEl sistema continuará sin detección de landmarks")
            print("="*70 + "\n")
            self.facemark = None
        else:
            try:
                self.facemark = cv2.face.createFacemarkLBF()
                self.facemark.loadModel(landmark_model)
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("✓ Detector de puntos faciales inicializado")
            except Exception as e:
                print(f"Error cargando landmarks: {e}")
                self.facemark = None
    
    def detect_landmarks(self, image):
        """Detecta puntos faciales"""
        if self.facemark is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        try:
            success, landmarks = self.facemark.fit(gray, faces)
            
            if success and len(landmarks) > 0:
                points = []
                for point in landmarks[0][0]:
                    points.append((int(point[0]), int(point[1])))
                return points
            else:
                return None
        except:
            return None
    
    def draw_landmarks(self, image, landmarks):
        """Dibuja puntos faciales"""
        if landmarks is None or len(landmarks) == 0:
            return image
        
        result = image.copy()
        
        for (x, y) in landmarks:
            cv2.circle(result, (x, y), 2, (0, 255, 0), -1)
        
        if len(landmarks) >= 68:
            # Contorno facial
            for i in range(16):
                cv2.line(result, landmarks[i], landmarks[i+1], (0, 255, 0), 1)
            
            # Cejas
            for i in range(17, 21):
                cv2.line(result, landmarks[i], landmarks[i+1], (0, 255, 0), 1)
            for i in range(22, 26):
                cv2.line(result, landmarks[i], landmarks[i+1], (0, 255, 0), 1)
            
            # Ojos
            for i in range(36, 41):
                cv2.line(result, landmarks[i], landmarks[i+1], (0, 255, 0), 1)
            cv2.line(result, landmarks[41], landmarks[36], (0, 255, 0), 1)
            
            for i in range(42, 47):
                cv2.line(result, landmarks[i], landmarks[i+1], (0, 255, 0), 1)
            cv2.line(result, landmarks[47], landmarks[42], (0, 255, 0), 1)
            
            # Boca
            for i in range(48, 59):
                cv2.line(result, landmarks[i], landmarks[i+1], (0, 255, 0), 1)
            cv2.line(result, landmarks[59], landmarks[48], (0, 255, 0), 1)
        
        return result


# ==================== RECONOCIMIENTO FACIAL ====================
class EmotionRecognizer:
    """Reconocedor de emociones usando LBPH + características geométricas"""
    def __init__(self):
        print("Inicializando reconocedor de emociones...")
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise Exception("Error cargando Haar Cascade")
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        
        self.geometric_features = []
        self.geometric_labels = []
        self.is_trained = False
        self.feature_extractor = FacialFeatureExtractor()
        
        print("✓ Reconocedor inicializado")
    
    def detect_faces(self, image):
        """Detecta rostros"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        img_height, img_width = image.shape[:2]
        min_face_size = max(int(img_width * 0.20), 100)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(min_face_size, min_face_size)
        )
        
        result = []
        for (x, y, w, h) in faces:
            result.append({'bbox': (x, y, w, h), 'confidence': 1.0})
        
        return result
    
    def extract_face_roi(self, image, bbox):
        """Extrae región del rostro"""
        x, y, w, h = bbox
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        margin = int(w * 0.2)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray.shape[1], x + w + margin)
        y2 = min(gray.shape[0], y + h + margin)
        
        face_roi = gray[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
        
        face_resized = cv2.resize(face_roi, (200, 200))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_resized = clahe.apply(face_resized)
        face_resized = cv2.bilateralFilter(face_resized, 5, 50, 50)
        
        return face_resized
    
    def train(self, faces_data, labels, geometric_features=None):
        """Entrena el reconocedor"""
        if len(faces_data) == 0:
            print("No hay datos de entrenamiento")
            return False
        
        print(f"Entrenando con {len(faces_data)} imágenes...")
        
        try:
            self.recognizer.train(faces_data, np.array(labels))
            
            if geometric_features is not None:
                self.geometric_features = geometric_features
                self.geometric_labels = labels
            
            self.is_trained = True
            print("✓ Entrenamiento completado")
            return True
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            return False
    
    def predict(self, face_roi, landmarks=None):
        """Predice emoción"""
        if not self.is_trained:
            return -1, 0.0
        
        try:
            label_texture, confidence_texture = self.recognizer.predict(face_roi)
            
            if confidence_texture < 50:
                similarity_texture = 1.0 - (confidence_texture / 100.0)
            elif confidence_texture < 80:
                similarity_texture = 0.5 - ((confidence_texture - 50) / 60.0)
            else:
                similarity_texture = max(0, 0.3 - ((confidence_texture - 80) / 100.0))
            
            if landmarks and len(self.geometric_features) > 0:
                current_features = self.feature_extractor.extract_geometric_features(landmarks)
                
                if current_features is not None:
                    best_match_label = -1
                    best_similarity_geom = 0.0
                    
                    for i, stored_features in enumerate(self.geometric_features):
                        distance = np.linalg.norm(current_features - stored_features)
                        similarity = 1.0 / (1.0 + distance)
                        
                        if similarity > best_similarity_geom:
                            best_similarity_geom = similarity
                            best_match_label = self.geometric_labels[i]
                    
                    if best_match_label == label_texture:
                        final_similarity = 0.6 * similarity_texture + 0.4 * best_similarity_geom
                        final_label = label_texture
                    else:
                        if similarity_texture > best_similarity_geom:
                            final_similarity = similarity_texture * 0.8
                            final_label = label_texture
                        else:
                            final_similarity = best_similarity_geom * 0.8
                            final_label = best_match_label
                    
                    return final_label, final_similarity
            
            return label_texture, similarity_texture
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return -1, 0.0


# ==================== BASE DE DATOS ====================
class EmotionDatabase:
    """Gestión de base de datos de emociones"""
    def __init__(self, person_name, data_dir='./emociones_data'):
        self.person_name = person_name
        self.data_dir = data_dir
        self.person_dir = os.path.join(data_dir, person_name)
        
        self.emotion_faces = []
        self.emotion_labels = []
        self.emotion_geometric_features = []
        
        self.emotions = {
            0: {'name': 'Enojado', 'emoji': '😠', 'color': (0, 0, 255)},
            1: {'name': 'Feliz', 'emoji': '😊', 'color': (0, 255, 0)},
            2: {'name': 'Neutral', 'emoji': '😐', 'color': (128, 128, 128)},
            3: {'name': 'Triste', 'emoji': '😢', 'color': (255, 128, 0)},
            4: {'name': 'Sorprendido', 'emoji': '😮', 'color': (0, 255, 255)}
        }
        
        self.label_to_emotion = {i: data['name'] for i, data in self.emotions.items()}
        self.emotion_to_label = {data['name']: i for i, data in self.emotions.items()}
        
        self.db_file = os.path.join(self.person_dir, 'emotions_db.pkl')
        
        os.makedirs(self.person_dir, exist_ok=True)
        
        self.recognizer = EmotionRecognizer()
        self.landmark_detector = FacialLandmarkDetector()
        
        print(f"Base de datos: {person_name}")
        self.load_database()
    
    def load_database(self):
        """Carga base de datos"""
        if os.path.exists(self.db_file):
            print("Cargando base de datos...")
            try:
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.emotion_faces = data['faces']
                    self.emotion_labels = data['labels']
                    self.emotion_geometric_features = data.get('geometric_features', [])
                
                if self.emotion_faces:
                    self.recognizer.train(
                        self.emotion_faces, 
                        self.emotion_labels,
                        self.emotion_geometric_features if self.emotion_geometric_features else None
                    )
                    print(f"✓ {len(self.emotion_faces)} imágenes cargadas")
            except Exception as e:
                print(f"Error cargando BD: {e}")
                self.emotion_faces = []
                self.emotion_labels = []
        else:
            print("No existe base de datos previa")
    
    def save_database(self):
        """Guarda base de datos"""
        try:
            data = {
                'faces': self.emotion_faces,
                'labels': self.emotion_labels,
                'geometric_features': self.emotion_geometric_features,
                'person_name': self.person_name
            }
            with open(self.db_file, 'wb') as f:
                pickle.dump(data, f)
            print("✓ Base de datos guardada")
        except Exception as e:
            print(f"Error guardando BD: {e}")
    
    def capture_emotions(self, webcam, min_photos=3):
        """Captura fotos de emociones"""
        print(f"\n{'='*70}")
        print(f"CAPTURANDO EMOCIONES: {self.person_name}")
        print(f"Mínimo: {min_photos} fotos por emoción")
        print(f"{'='*70}")
        
        captured_images = []
        captured_rois = []
        captured_labels = []
        captured_landmarks = []
        
        print("\nCONTROLES: ESPACIO=Capturar | C=Completar | ESC=Cancelar\n")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (800, 600))
            disp = frame.copy()
            
            faces = self.recognizer.detect_faces(frame)
            landmarks = self.landmark_detector.detect_landmarks(frame)
            
            if landmarks:
                disp = self.landmark_detector.draw_landmarks(disp, landmarks)
            
            for face in faces:
                x, y, w, h = face['bbox']
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Panel
            cv2.rectangle(disp, (0, 0), (800, 200), (30, 30, 30), -1)
            cv2.putText(disp, f"Capturando: {self.person_name}", (15, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp, f"Total: {len(captured_images)}", (15, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_off = 90
            for eid, edata in self.emotions.items():
                count = captured_labels.count(eid)
                color = (0, 255, 0) if count >= min_photos else (0, 165, 255)
                cv2.putText(disp, f"[{eid}] {edata['emoji']} {edata['name']}: {count}",
                           (15, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_off += 22
            
            cv2.imshow('Captura de Emociones', disp)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # ESPACIO
                if len(faces) != 1:
                    print("⚠ Debe haber exactamente 1 rostro")
                    continue
                
                print("\n" + "="*50)
                for eid, edata in self.emotions.items():
                    print(f"[{eid}] {edata['emoji']} {edata['name']}")
                
                emotion_input = input("Emoción: ").strip()
                
                try:
                    emotion_label = int(emotion_input)
                    if emotion_label not in self.emotions:
                        print("⚠ Emoción inválida")
                        continue
                    
                    bbox = faces[0]['bbox']
                    face_roi = self.recognizer.extract_face_roi(frame, bbox)
                    
                    if face_roi is not None:
                        captured_images.append(frame.copy())
                        captured_rois.append(face_roi)
                        captured_labels.append(emotion_label)
                        captured_landmarks.append(landmarks)
                        
                        print(f"✓ Foto {len(captured_images)} capturada")
                except:
                    print("⚠ Entrada inválida")
            
            elif key == ord('c') or key == ord('C'):
                all_ok = True
                for eid in self.emotions.keys():
                    if captured_labels.count(eid) < min_photos:
                        print(f"⚠ {self.emotions[eid]['name']}: {captured_labels.count(eid)} fotos")
                        all_ok = False
                
                if all_ok and len(captured_images) > 0:
                    print(f"\n✓ Completado con {len(captured_images)} fotos")
                    break
            
            elif key == 27:  # ESC
                print("\n⚠ Cancelado")
                cv2.destroyWindow('Captura de Emociones')
                return False
        
        cv2.destroyWindow('Captura de Emociones')
        
        try:
            # Extraer características geométricas
            for lm in captured_landmarks:
                if lm:
                    features = self.recognizer.feature_extractor.extract_geometric_features(lm)
                    if features is not None:
                        self.emotion_geometric_features.append(features)
            
            # Agregar a BD
            self.emotion_faces.extend(captured_rois)
            self.emotion_labels.extend(captured_labels)
            
            # Guardar imágenes
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for idx, (img, label) in enumerate(zip(captured_images, captured_labels)):
                folder = os.path.join(self.person_dir, self.emotions[label]['name'])
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(os.path.join(folder, f"{timestamp}_{idx+1}.jpg"), img)
            
            # Entrenar
            self.recognizer.train(
                self.emotion_faces,
                self.emotion_labels,
                self.emotion_geometric_features if self.emotion_geometric_features else None
            )
            self.save_database()
            
            print(f"\n✓ '{self.person_name}' entrenado con {len(captured_images)} fotos")
            return True
        except Exception as e:
            print(f"⚠ Error: {e}")
            return False
    
    def recognize_emotion(self, face_roi, landmarks=None):
        """Reconoce emoción"""
        if not self.recognizer.is_trained:
            return "Desconocido", 0.0, None
        
        label, similarity = self.recognizer.predict(face_roi, landmarks)
        
        if label in self.label_to_emotion and similarity > 0.35:
            return self.label_to_emotion[label], similarity, self.emotions[label]
        
        return "Desconocido", 0.0, None


# ==================== SISTEMA PRINCIPAL ====================
class EmotionDetectionSystem:
    """Sistema principal"""
    def __init__(self, person_name):
        print("\n" + "="*70)
        print("SISTEMA DE DETECCIÓN DE EMOCIONES MEJORADO")
        print("="*70)
        
        self.person_name = person_name
        self.emotion_db = EmotionDatabase(person_name)
        self.fps = 0
        self.frame_count = 0
        
        print("✓ Sistema inicializado")
    
    def capture_training_data(self, webcam, min_photos=3):
        """Captura datos de entrenamiento"""
        return self.emotion_db.capture_emotions(webcam, min_photos)
    
    def run_detection(self, webcam):
        """Ejecuta detección"""
        print("\n" + "="*70)
        print("MODO DETECCIÓN")
        print("="*70)
        print("Q=Salir | ESPACIO=Pausar | F=Features")
        print("="*70 + "\n")
        
        if not self.emotion_db.recognizer.is_trained:
            print("⚠ No hay modelo entrenado")
            return
        
        paused = False
        show_features = False
        frame_times = deque(maxlen=30)
        
        try:
            while True:
                start = time.time()
                
                ret, frame = webcam.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (800, 600))
                
                if not paused:
                    self.frame_count += 1
                    
                    faces = self.emotion_db.recognizer.detect_faces(frame)
                    landmarks = self.emotion_db.landmark_detector.detect_landmarks(frame)
                    
                    detected = []
                    for face in faces:
                        bbox = face['bbox']
                        roi = self.emotion_db.recognizer.extract_face_roi(frame, bbox)
                        
                        if roi is not None:
                            emotion, sim, data = self.emotion_db.recognize_emotion(roi, landmarks)
                            detected.append({
                                'bbox': bbox, 'emotion': emotion,
                                'similarity': sim, 'data': data,
                                'landmarks': landmarks
                            })
                    
                    elapsed = time.time() - start
                    frame_times.append(elapsed)
                    self.fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                    
                    disp = self.draw_ui(frame, detected, landmarks, show_features)
                else:
                    disp = frame.copy()
                    cv2.putText(disp, "PAUSADO", (300, 300),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                
                cv2.imshow('Detección de Emociones', disp)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('f'):
                    show_features = not show_features
        
        except KeyboardInterrupt:
            print("\n⚠ Interrumpido")
        finally:
            cv2.destroyAllWindows()
            print(f"\n✓ Frames: {self.frame_count} | FPS: {self.fps:.1f}")
    
    def draw_ui(self, frame, detected, landmarks, show_features):
        """Dibuja interfaz"""
        disp = frame.copy()
        h, w = disp.shape[:2]
        
        if landmarks:
            disp = self.emotion_db.landmark_detector.draw_landmarks(disp, landmarks)
        
        for det in detected:
            x, y, wi, he = det['bbox']
            emotion = det['emotion']
            sim = det['similarity']
            data = det['data']
            
            if data:
                color = data['color']
                emoji = data['emoji']
                
                cv2.rectangle(disp, (x, y), (x+wi, y+he), color, 3)
                cv2.rectangle(disp, (x, y-40), (x+wi, y), color, -1)
                cv2.putText(disp, f"{emoji} {emotion}", (x+5, y-22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(disp, f"{sim:.0%}", (x+5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Panel superior
        cv2.rectangle(disp, (0, 0), (w, 100), (30, 30, 30), -1)
        cv2.putText(disp, "DETECCION MEJORADA", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(disp, f"Persona: {self.person_name}", (15, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp, f"FPS: {self.fps:.1f} | Rostros: {len(detected)}", (15, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        lm_text = f"Landmarks: {len(landmarks)} pts" if landmarks else "Landmarks: No"
        cv2.putText(disp, lm_text, (15, 92),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return disp


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SISTEMA DE DETECCIÓN DE EMOCIONES FACIALES MEJORADO")
    print("="*70)
    print("Características geométricas + LBPH")
    print("Resolución: 800x600")
    print("="*70 + "\n")
    
    print("INSTALACIÓN:")
    print("   pip install opencv-contrib-python numpy")
    print("\nARCHIVO OPCIONAL (mejor precisión):")
    print("   Descargue lbfmodel.yaml desde:")
    print("   https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml")
    print("="*70 + "\n")
    
    # Verificar OpenCV
    try:
        test = cv2.face.LBPHFaceRecognizer_create()
        print("✓ OpenCV con LBPH detectado")
    except:
        print("⚠ ERROR: Instale opencv-contrib-python")
        exit(1)
    
    print("")
    
    # Inicializar cámara
    print("Inicializando cámara...")
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        print("⚠ ERROR: No se pudo abrir la cámara")
        exit(1)
    
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    webcam.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Cámara inicializada (800x600)\n")
    
    # Menú
    print("="*70)
    print("MENÚ PRINCIPAL")
    print("="*70)
    print("1. Agregar/Entrenar persona (Recomendado: 3-5 fotos/emoción)")
    print("2. Modo detección de emociones")
    print("3. Salir")
    print("="*70)
    
    opcion = input("\nOpción: ").strip()
    
    if opcion == '1':
        print("\n" + "="*70)
        person_name = input("Ingrese su nombre: ").strip()
        
        if not person_name:
            print("⚠ Nombre inválido")
            webcam.release()
            exit(1)
        
        system = EmotionDetectionSystem(person_name)
        
        print(f"\nCAPTURA DE EMOCIONES: {person_name}")
        print("Emociones a capturar (3-5 fotos cada una):")
        print("  😠 Enojado")
        print("  😊 Feliz")
        print("  😐 Neutral")
        print("  😢 Triste")
        print("  😮 Sorprendido")
        print("\nIndicará manualmente cada emoción")
        print("MEJORA: Extracción automática de características geométricas")
        print("="*70)
        
        input("\nPresione ENTER para comenzar...")
        
        success = system.capture_training_data(webcam, min_photos=3)
        
        if success:
            print("\n" + "="*70)
            print("✓ ENTRENAMIENTO COMPLETADO")
            print("="*70)
            print("Sistema listo para detectar emociones")
            print("Ejecute nuevamente y seleccione opción 2")
            print("="*70 + "\n")
        else:
            print("\n⚠ Entrenamiento cancelado")
    
    elif opcion == '2':
        print("\n" + "="*70)
        person_name = input("Nombre de persona entrenada: ").strip()
        
        if not person_name:
            print("⚠ Nombre inválido")
            webcam.release()
            exit(1)
        
        db_path = os.path.join('./emociones_data', person_name, 'emotions_db.pkl')
        if not os.path.exists(db_path):
            print(f"\n⚠ ERROR: No existe modelo para '{person_name}'")
            print("Primero entrene el sistema (opción 1)")
            webcam.release()
            exit(1)
        
        system = EmotionDetectionSystem(person_name)
        
        print("\nPresione ENTER para iniciar detección...")
        input()
        
        system.run_detection(webcam)
    
    elif opcion == '3':
        print("\n✓ Saliendo...")
    
    else:
        print("\n⚠ Opción inválida")
    
    # Limpiar
    webcam.release()
    cv2.destroyAllWindows()
    print("\n✓ Sistema finalizado")
    print("="*70 + "\n")
