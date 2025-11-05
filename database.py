import sqlite3
import json
from datetime import datetime
import os
import numpy as np

class PredictionDatabase:
    def __init__(self, db_path="predictions.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                upload_date TIMESTAMP,
                image_path TEXT,
                tumor_present BOOLEAN,
                predicted_grade INTEGER,
                grade_confidence REAL,
                tumor_area REAL,
                results_json TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_prediction(self, patient_name, image_path, results):
        # Save tumor_mask separately
        mask_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.npy"
        mask_path = os.path.join("static", "masks", mask_filename)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        np.save(mask_path, results['tumor_mask'])
        
        # Replace array in results with path
        serializable_results = results.copy()
        serializable_results['tumor_mask'] = mask_path
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(''' 
            INSERT INTO predictions 
            (patient_name, upload_date, image_path, tumor_present, predicted_grade, 
             grade_confidence, tumor_area, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_name,
            datetime.now(),
            image_path,
            results['tumor_present'],
            results['predicted_grade'],
            results['grade_confidence'],
            results['tumor_area'],
            json.dumps(serializable_results)
        ))
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        return prediction_id


    
    def get_all_predictions(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM predictions ORDER BY upload_date DESC')
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_prediction(self, prediction_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
        result = cursor.fetchone()
        conn.close()
        return result
