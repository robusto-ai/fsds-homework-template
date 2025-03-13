### Homework 1: Introduction to BentoML with a Dummy Model

#### Requirements and Description

This homework introduces you to BentoML, a framework for serving machine learning models. You’ll build a simple anomaly detection service using a dummy model (no external libraries like `alibi-detect` required). The focus is on understanding BentoML’s core components: runners, services, and basic inference.

You will implement three functions:

1. **Exercise 1: Define a Prediction Runner**
   - **Task**: Create a `PredictionRunner` class (not `AnomalyDetectionRunner`) that inherits from `bentoml.Runnable`. This runner will simulate anomaly detection using a simple rule: if the sum of input values exceeds a threshold (set to 10.0), classify it as an anomaly (1), otherwise not (0).
   - **Requirements**:
     - Set `SUPPORTED_RESOURCES = ("cpu",)` and `SUPPORTS_CPU_MULTI_THREADING = True`.
     - Define `__init__` to set the threshold.
     - Implement an `inference` method (marked with `@bentoml.Runnable.method(batchable=False)`) that takes a numpy array and returns a list of 0s and 1s.
   - **Example**:
     ```bash
     >>> runner = PredictionRunner()
     >>> runner.inference(np.array([2.0, 3.0]))
     [0]  # Sum = 5.0 < 10.0
     >>> runner.inference(np.array([5.0, 6.0]))
     [1]  # Sum = 11.0 > 10.0
     ```

2. **Exercise 2: Create a BentoML Service**
   - **Task**: Write a function `create_prediction_service` that initializes a BentoML service with a `PredictionRunner`.
   - **Requirements**:
     - Create a runner instance using `bentoml.Runner`.
     - Define a service using `bentoml.Service` with the name `"simple_anomaly_detection"` and the runner.
     - Return the service object.
   - **Example**:
     ```bash
     >>> svc = create_prediction_service()
     >>> svc.name
     'simple_anomaly_detection'
     >>> len(svc.runners)
     1
     ```

3. **Exercise 3: Implement a Simple Inference API**
   - **Task**: Write a function `run_simple_inference` that takes a numpy array input, runs it through the service’s runner, and returns the prediction.
   - **Requirements**:
     - Use the runner’s `inference.run()` method (synchronous call).
     - Handle basic input validation: raise a `ValueError` if the input is not a numpy array.
   - **Example**:
     ```bash
     >>> run_simple_inference(np.array([1.0, 2.0]))
     [0]  # Sum = 3.0 < 10.0
     >>> run_simple_inference(np.array([7.0, 8.0]))
     [1]  # Sum = 15.0 > 10.0
     >>> run_simple_inference(None)
     ValueError: Input must be a numpy array
     ```