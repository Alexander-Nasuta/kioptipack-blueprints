import mlflow

import mlflow.pytorch
import torch

import pandas as pd
import numpy as np

MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


if __name__ == '__main__':
    model_uri = "models:/MyModel/4"
    model = mlflow.pytorch.load_model(model_uri=model_uri)

    print(model)

    data = [{
        'aufbereiteter_wert': 0.287696,
        'laborant_AN': 0.0,
        'laborant_HANS': 0.0,
        'laborant_SO': 0.0,
        'laborant_TK': 1.0,
        'material_id_00000000': 0.0,
        'material_id_11111111': 1.0,
        'material_id_22222222': 0.0,
        'material_id_33333333': 0.0,
        'rohwert_1_high': 0.0,
        'rohwert_1_low': 0.0,
        'rohwert_1_medium': 0.0,
        'rohwert_1_very_high': 1.0,
        'rohwert_1_very_low': 0.0,
        'rohwert_2_labormessung': 0.55194,
        'rohwert_3_labormessung': -0.472279
    }] * 13

    temp = pd.DataFrame(data)

    y_data = np.array([temp.pop("aufbereiteter_wert")])
    x_data = temp.to_numpy()

    # do a prediction with the model and the data
    prediction = model(torch.tensor(x_data, dtype=torch.float32))
    # prediction to list
    prediction = prediction.tolist()
    print(prediction)
