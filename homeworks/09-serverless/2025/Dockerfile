FROM agrigorev/model-2025-hairstyle:v1

RUN pip install numpy pillow onnxruntime

COPY lambda_function.py .

ENTRYPOINT ["python"]

CMD ["lambda_function.py"]
