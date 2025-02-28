import bentoml
import numpy as np

# Exercise 1: Define a Prediction Runner
class PredictionRunner(bentoml.Runnable):
    # Define supported resources and multi-threading support
    SUPPORTED_RESOURCES = None  # TODO: Fill this in
    SUPPORTS_CPU_MULTI_THREADING = None  # TODO: Fill this in

    def __init__(self):
        # TODO: Set a threshold for anomaly detection
        pass

    @bentoml.Runnable.method(batchable=False)
    def inference(self, input_data):
        # TODO: Implement dummy anomaly detection
        # Return [1] if sum of input_data > threshold, else [0]
        pass

# Exercise 2: Create a BentoML Service
def create_prediction_service():
    """
    Create a BentoML service with a PredictionRunner.

    Returns:
        bentoml.Service: The initialized service.
    """
    # TODO: Create a runner and service, then return the service
    pass

# Exercise 3: Implement a Simple Inference API
def run_simple_inference(input_data):
    """
    Run inference on input data using the PredictionRunner.

    Args:
        input_data (np.ndarray): Input data as a numpy array.

    Returns:
        list: Prediction result ([0] or [1]).

    Raises:
        ValueError: If input_data is not a numpy array.
    """
    # TODO: Validate input and run inference
    pass