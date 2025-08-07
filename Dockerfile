FROM nvcr.io/nvidia/tritonserver:25.07-pyt-python-py3
RUN pip install pillow
CMD ["tritonserver", "--model-repository=/models"]