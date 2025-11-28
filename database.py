import json
import os
import numpy as np
from datetime import datetime
import streamlit as st
from streamlit_gsheets import GSheetsConnection

class PredictionDatabase:
    def __init__(self, sheet_name="predictions"):
        """Initialize Google Sheets connection"""
        self.sheet_name = sheet_name
        self.conn = st.connection("gsheets", type=GSheetsConnection)
        self._create_sheet_if_missing()

    def _create_sheet_if_missing(self):
        """Ensure the sheet has required structure"""
        df = self.conn.read(worksheet=self.sheet_name)

        if df is None or df.empty:
            df = self._empty_dataframe()
            self.conn.update(worksheet=self.sheet_name, data=df)

    def _empty_dataframe(self):
        """Return empty dataframe with proper columns"""
        import pandas as pd
        return pd.DataFrame({
            "id": [],
            "patient_name": [],
            "upload_date": [],
            "image_path": [],
            "tumor_present": [],
            "predicted_grade": [],
            "grade_confidence": [],
            "tumor_area": [],
            "results_json": []
        })

    def _next_id(self, df):
        """Generate incremental ID"""
        if df["id"].empty:
            return 1
        return int(df["id"].max()) + 1

    def save_prediction(self, patient_name, image_path, results):
        """Save prediction into Google Sheets"""

        # Save tumor mask
        mask_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.npy"
        mask_dir = os.path.join("static", "masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, mask_filename)
        np.save(mask_path, results["tumor_mask"])

        # Make results JSON serializable
        serializable_results = results.copy()
        serializable_results["tumor_mask"] = mask_path

        # Load sheet
        df = self.conn.read(worksheet=self.sheet_name)

        # Assign new ID
        new_id = self._next_id(df)

        # Create new row
        new_row = {
            "id": new_id,
            "patient_name": patient_name,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": image_path,
            "tumor_present": int(results["tumor_present"]),
            "predicted_grade": int(results["predicted_grade"]),
            "grade_confidence": float(results["grade_confidence"]),
            "tumor_area": float(results["tumor_area"]),
            "results_json": json.dumps(serializable_results)
        }

        # Append
        df = df.append(new_row, ignore_index=True)

        # Save back to sheet
        self.conn.update(worksheet=self.sheet_name, data=df)

        return new_id

    def get_all_predictions(self):
        """Retrieve all predictions"""
        df = self.conn.read(worksheet=self.sheet_name)
        if df is None or df.empty:
            return []
        return df.to_dict("records")

    def get_prediction(self, prediction_id):
        """Retrieve single prediction"""
        df = self.conn.read(worksheet=self.sheet_name)
        row = df[df["id"] == int(prediction_id)]
        if row.empty:
            return None
        return row.to_dict("records")[0]

    def delete_prediction(self, prediction_id):
        """Delete a prediction"""
        df = self.conn.read(worksheet=self.sheet_name)

        row = df[df["id"] == int(prediction_id)]
        if row.empty:
            return False

        # Delete physical mask file
        try:
            results = json.loads(row.iloc[0]["results_json"])
            mask_path = results.get("tumor_mask")
            if mask_path and os.path.exists(mask_path):
                os.remove(mask_path)
        except:
            pass

        # Remove from sheet
        df = df[df["id"] != int(prediction_id)]
        self.conn.update(worksheet=self.sheet_name, data=df)

        return True

    def get_statistics(self):
        """Compute statistics"""
        df = self.conn.read(worksheet=self.sheet_name)

        if df.empty:
            return {
                "total_predictions": 0,
                "tumor_count": 0,
                "no_tumor_count": 0,
                "avg_confidence": 0,
                "avg_grade": 0
            }

        total = len(df)
        tumor_count = df[df["tumor_present"] == 1].shape[0]
        tumor_df = df[df["tumor_present"] == 1]

        avg_conf = tumor_df["grade_confidence"].mean() if not tumor_df.empty else 0
        avg_grade = tumor_df["predicted_grade"].mean() if not tumor_df.empty else 0

        return {
            "total_predictions": total,
            "tumor_count": tumor_count,
            "no_tumor_count": total - tumor_count,
            "avg_confidence": avg_conf,
            "avg_grade": avg_grade
        }
