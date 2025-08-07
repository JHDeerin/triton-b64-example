import base64
import io
import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        print(len(requests))
        for request in requests:
            # Get the input tensor as a numpy object
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE_B64")
            b64_str = in_tensor.as_numpy()[0].decode('utf-8')  # string from TYPE_STRING

            # Decode base64 -> JPEG bytes
            img_bytes = base64.b64decode(b64_str)

            # Load image via Pillow
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Create output tensor
            img_array = np.asarray(img, dtype=np.float32)
            out_tensor = pb_utils.Tensor("OUTPUT_IMAGE_TENSOR", img_array)

            # Build inference response
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
