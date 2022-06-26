import requests
import sys

# configure the host here
# API_HOST = "calcium.int.contriveit.com:5000"
API_HOST = "localhost:5000"

API_URL = "http://" + API_HOST + "/predict"

def predict_result(image_path):
    print('predict_result: start')
    image = open(image_path, 'rb').read()
    print('predict_result: image read')
    payload = {'image': image}
    print('predict_result: payload created')
    r = requests.post(API_URL, files=payload).json()
    print('predict_result: returned r')
    print('predict_result: end')
    return r

img_path = sys.argv[1]
print("Checking results for {}".format(img_path))
result = predict_result(img_path)
print(result)
