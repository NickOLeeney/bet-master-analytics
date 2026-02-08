import joblib
import pandas as pd
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession
from sklearn.compose import ColumnTransformer

def bool_to_int(x: pd.DataFrame):
    xc = x.copy()
    for col in xc.columns:
        xc[col] = xc[col].astype("int8")
    return xc


def datetime_to_int64(x: pd.DataFrame):
    """
    Converte datetime -> int64 (nanosecondi dal 1970). Evita feature engineering extra.
    """
    xc = x.copy()
    for col in xc.columns:
        # pandas datetime64[ns]
        xc[col] = pd.to_datetime(xc[col], errors="coerce").astype("int64")
    return xc


def cast_float64(x):
    return x.astype("float64")


def bool_to_int_df(x):
    return bool_to_int(pd.DataFrame(x, columns=bool_cols))


def datetime_to_int64_df(x):
    return datetime_to_int64(pd.DataFrame(x, columns=datetime_cols))


def get_onnx_prediction(sess: InferenceSession, preprocessor: ColumnTransformer, input: pd.DataFrame, threshold: float=0.5):
    # Preprocess input
    x_trans = preprocessor.transform(input)  # trasformazione del preprocessor salvato
    x_trans = x_trans.astype(np.float32)        # ONNX richiede float32
    # ONNX Inference
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x_trans})

    # Controlla quanti output ci sono e seleziona probabilità
    y_prob = np.array([x[1] for x in outputs[1]])  # se ONNX produce [label, probability]
    y_pred = np.array([int(x > threshold) for x in y_prob])

    return y_prob, y_pred


def main():
    onnx_export_path = "artifacts/model.onnx"
    preprocessor_export_path = "artifacts/preprocessor.joblib"

    # 1. Carica il modello ONNX
    sess = ort.InferenceSession(onnx_export_path)

    # 2. Carica il preprocessor (su Raspberry Pi)
    preprocessor = joblib.load(preprocessor_export_path)

    # 3. Esempio dataframe di input
    df_input = pd.DataFrame([{'chance1x2_quote_diffRealCurr1': -1.1,
                              'chance1x2_quote_diffRealCurr2': -8.8,
                              'evaluation_val1x2': 917006,
                              'evaluation_valScala': 49657,
                              'evaluation_valMetrica': 398418,
                              'chance1x2_bookkeeping_status': 1,
                              'chance1x2_quote_diffInitialCurr2': -5.2,
                              'chance1x2_quote_diffInitialCurr1': 0.0,
                              'chance1x2_quote_current1': 2.75},
                             {'chance1x2_quote_diffRealCurr1': 3.4,
                              'chance1x2_quote_diffRealCurr2': -6.7,
                              'evaluation_val1x2': 918177,
                              'evaluation_valScala': 48727,
                              'evaluation_valMetrica': 221599,
                              'chance1x2_bookkeeping_status': 0,
                              'chance1x2_quote_diffInitialCurr2': 16.7,
                              'chance1x2_quote_diffInitialCurr1': -15.6,
                              'chance1x2_quote_current1': 2.37},
                             {'chance1x2_quote_diffRealCurr1': 27.5,
                              'chance1x2_quote_diffRealCurr2': -23.3,
                              'evaluation_val1x2': 916968,
                              'evaluation_valScala': None,
                              'evaluation_valMetrica': 398418,
                              'chance1x2_bookkeeping_status': 0,
                              'chance1x2_quote_diffInitialCurr2': -14.8,
                              'chance1x2_quote_diffInitialCurr1': 21.6,
                              'chance1x2_quote_current1': 2.55},
                             {'chance1x2_quote_diffRealCurr1': 9.8,
                              'chance1x2_quote_diffRealCurr2': -13.8,
                              'evaluation_val1x2': 224234,
                              'evaluation_valScala': 49867,
                              'evaluation_valMetrica': None,
                              'chance1x2_bookkeeping_status': 0,
                              'chance1x2_quote_diffInitialCurr2': 0.0,
                              'chance1x2_quote_diffInitialCurr1': -7.1,
                              'chance1x2_quote_current1': 1.83},
                             {'chance1x2_quote_diffRealCurr1': 41.6,
                              'chance1x2_quote_diffRealCurr2': -55.3,
                              'evaluation_val1x2': 247917,
                              'evaluation_valScala': 49867,
                              'evaluation_valMetrica': None,
                              'chance1x2_bookkeeping_status': 3,
                              'chance1x2_quote_diffInitialCurr2': -5.3,
                              'chance1x2_quote_diffInitialCurr1': 6.4,
                              'chance1x2_quote_current1': 1.25}])

    threshold=0.875
    y_prob, y_pred = get_onnx_prediction(sess, preprocessor, df_input, threshold=threshold)
    print(f"Soglia scelta per il modello: {threshold}")
    print(f"Predizioni modello: {y_pred}")
    print(f"Probabilità modello: {y_prob}")
if __name__ == "__main__":
    main()