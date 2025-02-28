import bentoml
import numpy as np

class PredictionRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.threshold = 10.0

    @bentoml.Runnable.method(batchable=False)
    def inference(self, input_data):
        total = np.sum(input_data)
        return [1] if total > self.threshold else [0]

def create_prediction_service():
    runner = bentoml.Runner(PredictionRunner)
    svc = bentoml.Service("simple_anomaly_detection", runners=[runner])
    return svc

def run_simple_inference(input_data):
    if not isinstance(input_data, np.ndarray):
        raise ValueError("Input must be a numpy array")
    runner = bentoml.Runner(PredictionRunner)
    return runner.inference.run(input_data)