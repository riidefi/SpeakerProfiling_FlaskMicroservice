# SpeakerProfiling_FlaskMicroservice
Flask Microservice implementing https://arxiv.org/abs/2203.11774 on-demand

Fork of `SpeakerProfiling`. Download the weights (checkpoint file) and place it in a `model_checkpoint` folder to prepare the model for deployment.

### API Docs
This service provides an API endpoint to predict age, height, and gender from raw PCM audio data. It's built using a Flask server and utilizes a machine learning model trained on the TIMIT dataset.

## Running the Server

To start the server, run the Python script containing the Flask application.

```bash
python flask_server.py
```

This will start the server on `http://127.0.0.1:5000/` by default.

## API Endpoint

### Predict Age, Height, and Gender

- **URL**

  `/predict`

- **Method:**

  `POST`
  
- **Data Params**

  **Required:**

  `audio=[raw PCM audio data]`

- **Success Response:**

  - **Code:** 200 <br />
    **Content:** 
    ```json
    {
        "age": "predicted_age",
        "height": "predicted_height",
        "gender_is_female": "predicted_gender"
    }
    ```
 
- **Error Response:**

  - **Code:** 400 BAD REQUEST <br />
    **Content:** `{ error : "Error message" }`

- **Sample Call:**

  ```bash
  curl -X POST -H "Content-Type: application/octet-stream" --data-binary @your_audio_file.raw http://127.0.0.1:5000/predict
  ```

## Sample client
See `flask_test.py` for an example client.

## Notes

- The endpoint expects raw PCM audio data with a sample rate of 16000 Hz.
- Ensure that your audio file is in the correct format before sending it to the server.
- The predictions (age, height, and gender) are based on the model trained on the TIMIT dataset and may vary based on the quality and characteristics of the input audio.

## License

MIT License


Original README follows
# Speaker Profiling

This Repository contains the code for estimating the Age and Height of a speaker with their speech signal. This repository uses [s3prl](https://github.com/s3prl/s3prl) library to load various upstream models like wav2vec2, CPC, TERA etc. This repository uses TIMIT dataset. 

**_NOTE:_**  If you want to run the single encoder model, you should checkout the `singleEncoder` branch and follow the README in that branch.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages for preparing the dataset, training and testing the model.

```bash
pip install -r requirements.txt
```

## Usage

### Download the TIMIT dataset
```bash
wget https://data.deepai.org/timit.zip
unzip timit.zip -d 'path to timit data folder'
```

### Prepare the dataset for training and testing
```bash
python TIMIT/prepare_timit_data.py --path='path to timit data folder'
```

### Update Config and Logger
Update the config.json file to update the upstream model, batch_size, gpus, lr, etc and change the preferred logger in train_.py files. Create a folder 'checkpoints' to save the best models. If you wish to perform narrow band experiment, just set narrow_band as true in config.json file.

### Training
```bash
python train_timit.py --data_path='path to final data folder' --speaker_csv_path='path to this repo/SpeakerProfiling/Dataset/data_info_height_age.csv'
```

Example:
```bash
python train_timit.py --data_path=/notebooks/SpeakerProfiling/TIMIT_Dataset/wav_data/ --speaker_csv_path=/notebooks/SpeakerProfiling/Dataset/data_info_height_age.csv
```

### Testing
```bash
python test_timit.py --data_path='path to final data folder' --model_checkpoint='path to saved model checkpoint'
```

Example:
```bash
python test_timit.py --data_path=/notebooks/SpeakerProfiling/TIMIT_Dataset/wav_data/ --model_checkpoint=checkpoints/epoch=1-step=245-v3.ckpt
```

### Pretrained Model
We have uploaded a pretrained model of our experiments. You can download the from [Dropbox](https://www.dropbox.com/s/e9juyocxgigvekl/epoch%3D24-step%3D12249.ckpt?dl=0).

Download it and put it into the model_checkpoint folder.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S3prl: The self-supervised speech pre-training and representation learning toolkit. AT Liu, Y Shu-wen

