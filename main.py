import base64
import numpy as np
import requests

TRITON_URL = "http://localhost:8000/v2/models/image_decode/infer"

def encode_image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def send_request(image_path):
    # Encode to base64 string
    b64_str = encode_image_to_base64(image_path)
    print("Raw b64:")
    print(b64_str)

    # Triton expects the string in a numpy array for TYPE_STRING
    inputs = [
        {
            "name": "INPUT_IMAGE_B64",
            "shape": [1],
            "datatype": "BYTES",
            "data": [b64_str]
        }
    ]

    outputs = [{"name": "OUTPUT_IMAGE_TENSOR"}]

    payload = {
        "inputs": inputs,
        "outputs": outputs
    }

    r = requests.post(TRITON_URL, json=payload)
    r.raise_for_status()
    result = r.json()
    print("Raw JSON response:")
    print(result)

    # Triton returns outputs in base64 for BYTES or nested lists for numeric tensors
    raw_data = result['outputs'][0]['data']
    shape = result['outputs'][0]['shape']
    np_array = np.array(raw_data, dtype=np.float32).reshape(shape)

    return np_array

if __name__ == "__main__":
    img_tensor = send_request("test.jpeg")
    print("Image tensor shape:", img_tensor.shape)
    print("Full tensor:", img_tensor)
