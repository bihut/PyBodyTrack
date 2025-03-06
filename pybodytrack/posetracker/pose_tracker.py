import sys

import pandas as pd
import numpy as np
import time

class PoseTracker:
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


    def process_frame(self, frame):
        """Process a single frame and store the results."""
        pose_data, _ = self.processor.process(frame,selected_landmarks=self.selected_landmarks)

        # 📌 Asegurar que solo se guarden los landmarks correctos
        cleaned_data = {"timestamp": time.time()}

        for landmark in self.selected_landmarks:
            cleaned_data[landmark] = pose_data.get(landmark, (np.nan, np.nan, 0, np.nan))  # 📌 Z = 0 para YOLO

        if self.data is None:
            print("ERA NONE")
            self.data = []

        self.data.append(cleaned_data)
        print("LEN ", len(self.data))
        print("LEN COLS:",len(cleaned_data))
    def get_dataframe(self):
        """Returns a DataFrame with properly formatted column names."""
        if not self.data or self.data is None:
            print("❌ No hay datos en self.data. Algo está fallando en la recolección de datos.")
            return pd.DataFrame()

        error=0
        lista = None
        # 📌 Convertir lista de diccionarios a DataFrame
        try:
            error=1
            print("DATA TYPE",type(self.data))
            df = pd.DataFrame(self.data)
            lista = self.data

            # 📌 Separar los valores de los landmarks en columnas individuales
            landmark_dfs = []
            for landmark in self.selected_landmarks:
                error=2
                if landmark in df:
                    error=3
                    landmark_df = pd.DataFrame(df[landmark].tolist(), columns=[
                        f"{landmark}_x", f"{landmark}_y", f"{landmark}_z", f"{landmark}_confidence"
                    ])
                    error=4
                    landmark_dfs.append(landmark_df)
                else:
                    error=5
                    empty_df = pd.DataFrame(np.nan, index=df.index, columns=[
                        f"{landmark}_x", f"{landmark}_y", f"{landmark}_z", f"{landmark}_confidence"
                    ])
                    error=6
                    landmark_dfs.append(empty_df)

            # 📌 Concatenar todas las columnas en un solo DataFrame optimizado
            error =7
            df_final = pd.concat([df[["timestamp"]]] + landmark_dfs, axis=1)

            return df_final
        except Exception as e:
            print("Ha fallado:", str(e))
            print("ERROR:",error)
            print(type(lista))  # Verifica que sea una lista o un diccionario válido
            #print(len(lista))  # Verifica la longitud
            if isinstance(lista, list):
                print([len(item) for item in lista if
                       isinstance(item, list)])  # Verifica la longitud de las listas internas

            #sys.exit(0)

    def save_to_csv(self, filename="pose_data.csv"):
        """Guarda el DataFrame en un archivo CSV."""
        df = self.get_dataframe()
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"✅ Datos guardados en {filename}")
        else:
            print("❌ No se generaron datos en el DataFrame.")
