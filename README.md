# coffeacasa-triton-test
Example of using inference with NVIDIA Triton server with PyTorch model.

`binary_classifier_client.py` gives a simple example of using inference with the gRPC client. This can be run using the command

```
python binary_classifier_client.py -m binary_classifier testevents.csv -n 5
```

where `binary_classifier` is the model and `testevents.csv` is a csv file which contains the input data. `-n 5` means to run inference for 5 events.

`binary_classifier_client.ipynb` gives an interactive notebook example of processing files using a coffea processor and performing inference within the 
processor. Dynamic batching is used for sending inference requests. For more documentation about using the NVIDIA Triton client, see below:

[https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
[https://github.com/triton-inference-server/client](https://github.com/triton-inference-server/client)

The model directory `binary_classifier/` is provided for reference as well, though this is not the copy which is loaded into the inference server. Instructions for modifying the model configuration can be found in `binary_classifier_client.ipynb`.
