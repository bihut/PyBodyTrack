import pandas as pd
import numpy as np
import time

class CameraPoseTracker:
    """Tracks human pose using different models and stores data in a DataFrame."""

    def __init__(self, processor, selected_landmarks=None):
        """
        Parameters:
            processor: El procesador de pose (ej. MediaPipe o YOLO).
            selected_landmarks (list, opcional): Lista de landmarks a procesar. Si se
                omite, se usarán todos los landmarks estándar obtenidos del procesador.
        """
        self.processor = processor
        self.data = []

        # Obtener los landmarks estándar del procesador
        self.STANDARD_LANDMARKS = self.processor.get_standard_landmarks()

        # Si se proporciona una lista de landmarks, se usa esa; de lo contrario se usan todos

        if selected_landmarks is not None:
            # Asegurarse de que los landmarks seleccionados estén dentro de los estándar
            self.selected_landmarks = [lm for lm in self.STANDARD_LANDMARKS if lm in selected_landmarks]
        else:
            self.selected_landmarks = self.STANDARD_LANDMARKS
        print("SELECTED LANDMARKS:", self.selected_landmarks)

    def process_frame(self, frame):
        """Process a single frame and store the results."""
        pose_data, _ = self.processor.process(frame,selected_landmarks=self.selected_landmarks)

        # 📌 Asegurar que solo se guarden los landmarks correctos
        cleaned_data = {"timestamp": time.time()}

        for landmark in self.selected_landmarks:
            cleaned_data[landmark] = pose_data.get(landmark, (np.nan, np.nan, 0, np.nan))  # 📌 Z = 0 para YOLO

        self.data.append(cleaned_data)

    def get_dataframe(self):
        """Returns a DataFrame with properly formatted column names."""
        if not self.data:
            print("❌ No hay datos en self.data. Algo está fallando en la recolección de datos.")
            return pd.DataFrame()

        #print("✅ Datos en self.data (primeros 5 frames):")
        #print(self.data[:5])

        # 📌 Convertir lista de diccionarios a DataFrame
        df = pd.DataFrame(self.data)

        # 📌 Separar los valores de los landmarks en columnas individuales
        landmark_dfs = []
        for landmark in self.selected_landmarks:
            if landmark in df:
                landmark_df = pd.DataFrame(df[landmark].tolist(), columns=[
                    f"{landmark}_x", f"{landmark}_y", f"{landmark}_z", f"{landmark}_confidence"
                ])
                landmark_dfs.append(landmark_df)
            else:
                empty_df = pd.DataFrame(np.nan, index=df.index, columns=[
                    f"{landmark}_x", f"{landmark}_y", f"{landmark}_z", f"{landmark}_confidence"
                ])
                landmark_dfs.append(empty_df)

        # 📌 Concatenar todas las columnas en un solo DataFrame optimizado
        df_final = pd.concat([df[["timestamp"]]] + landmark_dfs, axis=1)

        return df_final

    def save_to_csv(self, filename="pose_data.csv"):
        """Guarda el DataFrame en un archivo CSV."""
        df = self.get_dataframe()
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"✅ Datos guardados en {filename}")
        else:
            print("❌ No se generaron datos en el DataFrame.")
