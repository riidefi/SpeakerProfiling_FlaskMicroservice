import torch
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import pandas as pd
from TIMIT.lightning_model_uncertainty_loss import LightningModel
from config import TIMITConfig
import torch.nn.utils.rnn as rnn_utils

from argparse import ArgumentParser
import pytorch_lightning as pl

parser = ArgumentParser(add_help=True)
parser.add_argument('--data_path', type=str, default=TIMITConfig.data_path)
parser.add_argument('--speaker_csv_path', type=str, default=TIMITConfig.speaker_csv_path)
parser.add_argument('--batch_size', type=int, default=TIMITConfig.batch_size)
parser.add_argument('--epochs', type=int, default=TIMITConfig.epochs)
parser.add_argument('--num_layers', type=int, default=TIMITConfig.num_layers)
parser.add_argument('--feature_dim', type=int, default=TIMITConfig.feature_dim)
parser.add_argument('--lr', type=float, default=TIMITConfig.lr)
parser.add_argument('--gpu', type=int, default=TIMITConfig.gpu)
parser.add_argument('--n_workers', type=int, default=TIMITConfig.n_workers)
parser.add_argument('--dev', type=str, default=False)
parser.add_argument('--model_checkpoint', type=str, default=TIMITConfig.model_checkpoint)
parser.add_argument('--upstream_model', type=str, default=TIMITConfig.upstream_model)
parser.add_argument('--model_type', type=str, default=TIMITConfig.model_type)
parser.add_argument('--narrow_band', type=str, default=TIMITConfig.narrow_band)

parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

print("Loading model...")
# Load model
if hparams.model_checkpoint:
    model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
    model.to('cuda')
    model.eval()
else:
    raise Exception('Model checkpoint not found!')
print("...loaded")

def predict_demographics(audio, audio_length):
    # Convert to tensor and add batch dimension
    audio_tensor = audio.to("cuda")
    length_tensor = audio_length

    # Get predictions
    y_hat_h, y_hat_a, y_hat_g = model(audio_tensor, length_tensor)
    y_hat_h, y_hat_a, y_hat_g = y_hat_h.to('cpu'), y_hat_a.to('cpu'), y_hat_g.to('cpu')

    # Load statistics for normalization
    csv_path = TIMITConfig.speaker_csv_path
    df = pd.read_csv(csv_path)
    h_mean, h_std = df[df['Use'] == 'TRN']['height'].mean(), df[df['Use'] == 'TRN']['height'].std()
    a_mean, a_std = df[df['Use'] == 'TRN']['age'].mean(), df[df['Use'] == 'TRN']['age'].std()

    # Convert predictions to actual values
    height_pred = (y_hat_h * h_std + h_mean).item()
    age_pred = (y_hat_a * a_std + a_mean).item()
    gender_pred = 'Female' if y_hat_g > 0.5 else 'Male'

    return {'height': height_pred, 'age': age_pred, 'gender_is_female': float(y_hat_g)}

# Example usage
# raw_audio = [your raw PCM audio data here]
# predictions = predict_demographics(raw_audio)
# print(predictions)
from flask import Flask, request, jsonify
from flask_cors import CORS

import io
import torchaudio

app = Flask(__name__)
CORS(app)

@app.route('/v1/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'audio_data' not in request.files:
            return jsonify({"error": "No audio_data part in the request"}), 400

        audio_data = request.files['audio_data'].read()

        #with open("bruh.wav", 'wb') as f:
        #    f.write(audio_data.read())
        audio_io = io.BytesIO(audio_data)

        # Expected: 16000hz, mono
        audio, _ = torchaudio.load(audio_io)
        
        seql = [x.reshape(-1,) for x in audio]
        seq_length = [x.shape[0] for x in seql]
        data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
        response = predict_demographics(data, seq_length)

        return jsonify(response)


if __name__ == '__main__':
    app.run()