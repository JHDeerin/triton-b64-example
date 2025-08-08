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
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE_B64")
            # input for TYPE_STRING will be a numpy array with `bytes` inside
            in_array = in_tensor.as_numpy()

            out_tensor = self.batch_response(in_array)

            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def single_response(self, in_array: np.ndarray):
        """
        Use when configured max_batch_size = 0.

        Will return a tensor of size [Height, Width, 3].
        """
        # Create output tensor
        img_array = img_array = self._b64_array_to_img_array(in_array)
        out_tensor = pb_utils.Tensor("OUTPUT_IMAGE_TENSOR", img_array)
        return out_tensor

    def batch_response(self, in_array: np.ndarray):
        """
        Use when configured max_batch_size != 0.

        Will return a tensor of size [N, Height, Width, 3], where
        N="the number of batches in the request."
        """
        images = []
        for b64_batch in in_array:
            img_array = self._b64_array_to_img_array(b64_batch)
            images.append(img_array)
        all_images_array = np.array(images, dtype=np.float32)
        out_tensor = pb_utils.Tensor("OUTPUT_IMAGE_TENSOR", all_images_array)
        return out_tensor

    def _b64_array_to_img_array(self, b64_array: np.ndarray) -> np.ndarray:
        """
        Convert an input array w/ a base64 string to an RGB image array.

        Will return an array of size [Height, Width, 3] (i.e. "HWC" order).
        """
        b64_str = b64_array[0].decode("utf-8")
        img_bytes = base64.b64decode(b64_str)

        # Load image and convert it into a numpy array
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.asarray(img, dtype=np.float32)
        return img_array
