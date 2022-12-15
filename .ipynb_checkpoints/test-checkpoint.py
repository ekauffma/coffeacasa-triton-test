import argparse
import tritonclient
from tritonclient.utils import InferenceServerException
import tritonclient.grpc as grpcclient

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    
    FLAGS = parser.parse_args()
    
    try:
        triton_client = grpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
        print("triton client: ", triton_client)
        
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()
        
    model_name = 'simple'
    
    # print(triton_client.is_server_live(headers={'test': '1', 'dummy': '2'}))
    
    # Health
    # if not triton_client.is_server_live(headers={'test': '1', 'dummy': '2'}):
    #     print("FAILED : is_server_live")
    #     sys.exit(1)

#     if not triton_client.is_server_ready():
#         print("FAILED : is_server_ready")
#         sys.exit(1)

#     if not triton_client.is_model_ready(model_name):
#         print("FAILED : is_model_ready")
#         sys.exit(1)
    
    print(dir(triton_client))
    metadata = triton_client.get_server_metadata()
    if not (metadata.name == 'triton'):
        print("FAILED : get_server_metadata")
        sys.exit(1)
    print(metadata)