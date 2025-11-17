import os
import numpy as np
import logging
import joblib
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available: %s", e)

class MLQModel:
    def __init__(self, model_path=None, scaler_x_path=None, scaler_y_path=None):
        self.model = None
        self.model_path = model_path
        self.scaler_x = None
        self.scaler_y = None
        # Load scalers if provided
        if scaler_x_path and os.path.exists(scaler_x_path):
            try:
                self.scaler_x = joblib.load(scaler_x_path)
                logging.info("Loaded scaler_x from %s", scaler_x_path)
            except Exception as e:
                logging.warning("Failed to load scaler_x from %s: %s", scaler_x_path, e)
                self.scaler_x = None
        if scaler_y_path and os.path.exists(scaler_y_path):
            try:
                self.scaler_y = joblib.load(scaler_y_path)
                logging.info("Loaded scaler_y from %s", scaler_y_path)
            except Exception as e:
                logging.warning("Failed to load scaler_y from %s: %s", scaler_y_path, e)
                self.scaler_y = None

        if model_path and TF_AVAILABLE:
            try:
                self.model = load_model(model_path)
                logging.info("Loaded Keras model from %s", model_path)
            except Exception as e:
                logging.warning("Failed to load model from %s: %s", model_path, e)
                self.model = None
        else:
            if model_path and not TF_AVAILABLE:
                logging.warning("Model path provided but TensorFlow not available; using fallback.")
            else:
                logging.info("No model path provided; using fallback analytic model.")

    def predict_q(self, trace_width, frequency, R, Lg, Ll):
        """
        Predict Q factor. Input ordering for model/scaler is assumed to be:
        [frequency, R, Lg, Ll, trace_width]
        trace_width and frequency can be scalars or arrays; results will be
        broadcast to the appropriate shape.
        """
        tw = np.array(trace_width, dtype=float)
        fr = np.array(frequency, dtype=float)
        tw_b, fr_b = np.broadcast_arrays(tw, fr)

        n = tw_b.size
        samples = np.stack([
            fr_b.ravel(),
            np.full(n, R, dtype=float),
            np.full(n, Lg, dtype=float),
            np.full(n, Ll, dtype=float),
            tw_b.ravel()
        ], axis=1)

        if self.model is not None:
            try:
                X = samples
                if self.scaler_x is not None:
                    X = self.scaler_x.transform(X)
                preds = self.model.predict(X, verbose=0)
                preds = np.array(preds)
                # if scaler_y exists, inverse_transform
                if self.scaler_y is not None:
                    # ensure shape (n, 1)
                    preds_inv = self.scaler_y.inverse_transform(preds.reshape(-1, 1))
                    preds = preds_inv.ravel()
                else:
                    preds = preds.ravel()
                return preds.reshape(tw_b.shape)
            except Exception as e:
                logging.warning("Model prediction failed: %s. Using fallback.", e)
        return self._fallback_q(tw_b, fr_b, R, Lg, Ll)

    @staticmethod
    def _fallback_q(trace_width, frequency, R, Lg, Ll):
        q_tw = 415.0 - 9.5 * trace_width
        q_fr = -0.000012 * (frequency - 400.0) ** 2 + 132.0
        geom_factor = 1.0 - 0.02 * (R - 6.0) + 0.01 * (Ll - 10.0) - 0.015 * (Lg - 5.0)
        q_combined = np.minimum(q_tw, q_fr) * geom_factor
        return q_combined
