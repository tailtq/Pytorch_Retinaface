# detect + convert ONNX
python detect.py --network mobile0.25 --trained_model mobilenet0.25_Final.pth

# convert ONNX to Tensorflow
onnx-tf convert -i 240-model.onnx -o tf-240

# convert Tensorflow to Tensorflow Lite
python tflite_converter.py

# write tflite info
python model_writer.py --model_file 240-int8.tflite --label_file labelmap.txt --export_directory tflite-240