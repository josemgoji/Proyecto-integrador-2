import numpy as np
import pandas as pd
import sys
import os
import cloudpickle
import plotly.graph_objects as go



def load_datasets():
    current_dir = os.getcwd()
    ROOT_PATH = os.path.dirname(current_dir)
    sys.path.insert(1, ROOT_PATH)
    sys.path.insert(1, current_dir)
    import root

    train = pd.read_pickle(root.DIR_DATA_STAGE + 'train_preprocessed.pkl')
    sarimax = pd.read_pickle(root.DIR_DATA_ANALYTICS + 'LSTM_predictions_val.pkl')
    lgbm = pd.read_pickle(root.DIR_DATA_ANALYTICS + 'LGBM_predictions_val.pkl')
    lstm = pd.read_pickle(root.DIR_DATA_ANALYTICS + 'LSTM_predictions_val.pkl')
    return root, train, sarimax, lgbm, lstm


def load_pipeline():
    """Load the saved pipeline using cloudpickle."""
    import root
    with open(root.DIR_DATA_ANALYTICS + 'pipeline.pkl', 'rb') as f:
        pipeline = cloudpickle.load(f)
    return pipeline


def unscale_data(scaler, predictions):
    placeholder = np.zeros((len(predictions), 11))
    placeholder[:, 0] = predictions['target']
    predictions_scaled = scaler.inverse_transform(placeholder)[:, 0]
    predictions_scaled[predictions_scaled < 0] = 0
    predictions = pd.DataFrame(predictions_scaled, columns=predictions.columns, index=predictions.index)
    return predictions


def create_plots(root, val, predictions, name):
    # Gráfico de las predicciones vs valores reales en el conjunto de test del modelo con mejores parametros
    fig = go.Figure()
    trace1 = go.Scatter(x=val.index, y=val['target'], name="Real", mode="lines", line_color='#5F70EB')
    trace2 = go.Scatter(x=predictions.index, y=predictions['target'], name="Estimado", mode="lines", line_color="#4EA72E")
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Producción (kWh)",
        width=750,
        height=350,
        margin=dict(l=20, r=0, t=35, b=20),
        legend=dict(
            orientation="v",
            yanchor="top",
            xanchor="right",
            x=0.99,
            y=0.99
        )
    )
    fig.write_image(root.DIR_DATA_ANALYTICS + f'{name}_pred_vs_real.png', width=750, height=350)



def main():
    root, train, sarimax, lgbm, lstm = load_datasets()
    end_val = '2022-08-31 23:59:59'
    val = pd.DataFrame(train.loc[end_val:]['target'])

    pipeline = load_pipeline()
    scaler = pipeline['scale']

    val = unscale_data(scaler, val)
    sarimax = unscale_data(scaler, sarimax)
    lgbm = unscale_data(scaler, lgbm)
    lstm = unscale_data(scaler, lstm)

    # Test unscale a single value
    print(scaler.inverse_transform([[0.115978, 0,0,0,0,0,0,0,0,0,0]]))

    create_plots(root, val, sarimax, 'SARIMAX')
    create_plots(root, val, lgbm, 'LGBM')
    create_plots(root, val, lstm, 'LSTM')


if __name__ == "__main__":
    main()