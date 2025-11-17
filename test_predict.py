# test_predict.py - simple non-GUI verification for model_wrapper
from model_wrapper import MLQModel

if __name__ == "__main__":
    m = MLQModel()  # uses fallback if TensorFlow isn't available
    tw = [1.0, 2.0, 3.0]
    fr = 400.0
    R, Lg, Ll = 6.0, 5.0, 10.0
    pred = m.predict_q(tw, fr, R, Lg, Ll)
    print("prediction shape:", getattr(pred, 'shape', None))
    print(pred)
