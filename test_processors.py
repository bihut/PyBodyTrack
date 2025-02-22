
from pybodytrack.pose_estimators.camera_pose_tracker import CameraPoseTracker

tracker = CameraPoseTracker(processor_name="mediapipe_processor")
tracker.run()

# 📌 Obtener los datos en formato DataFrame
df = tracker.get_dataframe()

# 📌 Guardar en CSV con el formato correcto
print("VOY A GUADARLO")
csv_filename = "pose_tracking_data2.csv"
df.to_csv(csv_filename, index=False)

print(f"Datos guardados en {csv_filename}")
print(df.head())  # Muestra las primeras filas para verificar