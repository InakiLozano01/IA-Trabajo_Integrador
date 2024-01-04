import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        # Redimensionar el frame para una mayor velocidad
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Cargar imágenes de codificación desde la ruta especificada
        :param images_path:
        :return:
        """
        # Cargar imágenes
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} imágenes de codificación encontradas.".format(len(images_path)))

        # Almacenar la codificación y los nombres de las imágenes
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Obtener solo el nombre del archivo de la ruta inicial del archivo.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Obtener la codificación
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Almacenar el nombre del archivo y la codificación del archivo
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Imágenes de codificación cargadas")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Encontrar todas las caras y codificaciones de caras en el frame actual de video
        # Convertir la imagen de color BGR (que usa OpenCV) a color RGB (que usa face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Verificar si la cara es una coincidencia con la(s) cara(s) conocida(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Desconocido"

            # O en su lugar, usar la cara conocida con la distancia más pequeña a la nueva cara
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convertir a matriz numpy para ajustar las coordenadas con el redimensionado del frame rápidamente
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names