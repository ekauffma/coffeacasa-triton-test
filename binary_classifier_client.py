import argparse
import sys
import numpy as np

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


def parse_model(model_metadata, model_config):

    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))
        
    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)


    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")
                
    # Model input must have 1 dims
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 1 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                   len(input_metadata.shape)))
        
    n_features = input_metadata.shape[1 if input_batch_dim else 0]
    
    return (model_config.max_batch_size, input_metadata.name,
            output_metadata.name, n_features, input_config.format,
            input_metadata.datatype)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument(
        '-x',
        '--model-version',
        type=str,
        required=False,
        default="",
        help='Version of model. Default is to use latest version.')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('test_events',
                        type=str,
                        default=None,
                        help='Input csv file path containing lines with events to infer.')
    FLAGS = parser.parse_args()

    try: 
        # Create gRPC client for communicating with the server
        triton_client = grpcclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

#     # Make sure the model matches our requirements, and get some
#     # properties of the model that we need for preprocessing
    try:
        print("model_name = ", FLAGS.model_name)
        print("model_version = ", FLAGS.model_version)
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, 
            model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)
    
    model_config = model_config.config

    max_batch_size, input_name, output_name, n_features, format, dtype = parse_model(
        model_metadata, model_config)
    
    supports_batching = max_batch_size > 0
    if not supports_batching and FLAGS.batch_size != 1:
        print("ERROR: This model doesn't support batching.")
        sys.exit(1)

    # load csv into np array
    test_events = FLAGS.test_events
    data = np.loadtxt(test_events, dtype=np.float32, delimiter=',')
    
    print(data.shape)
    
    # batch data
    data_length = data.shape[0]
    num_batches = int(np.ceil(data_length/FLAGS.batch_size))
    
    startind = 0
    
    num_batches = 5
    
    for i in range(num_batches):
        data_current = data[startind:startind+FLAGS.batch_size,:]
        startind+=FLAGS.batch_size
        
        client = grpcclient

        inpt = [client.InferInput(input_name, data_current.shape, dtype)]
        inpt[0].set_data_from_numpy(data_current)

        output = client.InferRequestedOutput(output_name)
        
        results = triton_client.infer(model_name=FLAGS.model_name, 
                                      inputs=inpt, 
                                      outputs=[output])
    
        inference_output = results.as_numpy(output_name)
        print(np.round(inference_output))