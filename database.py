import sqlite3
import json
from datetime import datetime
import os
import numpy as np

class PredictionDatabase:
    def __init__(self, db_path="predictions.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Create predictions table if it doesn't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT NOT NULL,
                    upload_date TIMESTAMP NOT NULL,
                    image_path TEXT NOT NULL,
                    tumor_present BOOLEAN NOT NULL,
                    predicted_grade INTEGER NOT NULL,
                    grade_confidence REAL NOT NULL,
                    tumor_area REAL NOT NULL,
                    results_json TEXT NOT NULL
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            raise
    
    def save_prediction(self, patient_name, image_path, results):
        """Save prediction results to database"""
        try:
            # Save tumor_mask separately as numpy file
            mask_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.npy"
            mask_dir = os.path.join("static", "masks")
            os.makedirs(mask_dir, exist_ok=True)
            mask_path = os.path.join(mask_dir, mask_filename)
            np.save(mask_path, results['tumor_mask'])
            
            # Create serializable copy of results
            serializable_results = results.copy()
            serializable_results['tumor_mask'] = mask_path
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(''' 
                INSERT INTO predictions 
                (patient_name, upload_date, image_path, tumor_present, predicted_grade, 
                 grade_confidence, tumor_area, results_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_name,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                image_path,
                int(results['tumor_present']),
                int(results['predicted_grade']),
                float(results['grade_confidence']),
                float(results['tumor_area']),
                json.dumps(serializable_results)
            ))
            conn.commit()
            prediction_id = cursor.lastrowid
            conn.close()
            return prediction_id
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            raise
    
    def get_all_predictions(self):
        """Retrieve all predictions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM predictions ORDER BY upload_date DESC')
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            return []
    
    def get_prediction(self, prediction_id):
        """Retrieve a single prediction by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
            result = cursor.fetchone()
            conn.close()
            return result
        except Exception as e:
            print(f"Error getting prediction: {str(e)}")
            return None
    
    def delete_prediction(self, prediction_id):
        """Delete a prediction and its associated files"""
        try:
            # Get prediction to find associated files
            prediction = self.get_prediction(prediction_id)
            if prediction:
                results_json = json.loads(prediction[8])
                mask_path = results_json.get('tumor_mask')
                
                # Delete mask file if exists
                if mask_path and os.path.exists(mask_path):
                    os.remove(mask_path)
                
                # Delete from database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
                conn.commit()
                conn.close()
                return True
            return False
        except Exception as e:
            print(f"Error deleting prediction: {str(e)}")
            return False
    
    def get_statistics(self):
        """Get summary statistics from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) FROM predictions')
            total = cursor.fetchone()[0]
            
            # Tumor count
            cursor.execute('SELECT COUNT(*) FROM predictions WHERE tumor_present = 1')
            tumor_count = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute('SELECT AVG(grade_confidence) FROM predictions WHERE tumor_present = 1')
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Average grade
            cursor.execute('SELECT AVG(predicted_grade) FROM predictions WHERE tumor_present = 1')
            avg_grade = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_predictions': total,
                'tumor_count': tumor_count,
                'no_tumor_count': total - tumor_count,
                'avg_confidence': avg_confidence,
                'avg_grade': avg_grade
            }
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
            return None
