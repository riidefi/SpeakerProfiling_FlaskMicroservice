import requests
import json

class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def predict(self, wav_file_path):
        with open(wav_file_path, 'rb') as file:
            data = file.read()
        files = {'audio_data': data}
        response = requests.post(self.base_url + '/v1/predict', files=files)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code}")

def main():
    client = APIClient(base_url="http://127.0.0.1:5000")
    try:
        prediction = client.predict(wav_file_path="timit_unzipped/data/TEST/DR6/MDSC0/SA1.WAV.wav")
        print("Prediction:", prediction)
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()