# Audio Analysis Service

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
        "gender": "predicted_gender"
    }
    ```
 
- **Error Response:**

  - **Code:** 400 BAD REQUEST <br />
    **Content:** `{ error : "Error message" }`

- **Sample Call:**

  ```bash
  curl -X POST -H "Content-Type: application/octet-stream" --data-binary @your_audio_file.raw http://127.0.0.1:5000/predict
  ```

## Notes

- The endpoint expects raw PCM audio data with a sample rate of 16000 Hz.
- Ensure that your audio file is in the correct format before sending it to the server.
- The predictions (age, height, and gender) are based on the model trained on the TIMIT dataset and may vary based on the quality and characteristics of the input audio.

## License

MIT License
