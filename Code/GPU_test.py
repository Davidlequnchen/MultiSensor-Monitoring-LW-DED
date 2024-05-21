import tensorflow as tf
import torch
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
import time
import numpy as np
import lightgbm

def log_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0
    utilization = nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU Utilization: {utilization.gpu}%")
    print(f"GPU Memory Utilization: {utilization.memory}%")




def check_gpu_support():
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    # print(label)
    train_data = lightgbm.Dataset(data, label=label)
    params = {'num_iterations': 1, 'device': 'cuda'}
    try:
        gbm = lightgbm.train(params, train_set=train_data)
        print("GPU True !!!")
    except Exception as e:
        print("GPU False !!!")





def train_lightgbm_gpu():
    # Load dataset
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Define parameters, including GPU usage
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'device_type': 'cuda',  # Use GPU
        # 'gpu_platform_id': 0,  # Optional, specify GPU platform id
        # 'gpu_device_id': 0,    # Optional, specify GPU device id
        # 'verbose': -1,         # Suppress LightGBM verbose output
        # 'early_stopping_rounds': 10  # Add early stopping to params
    }

    # Log GPU utilization before training
    print("Before training:")
    log_gpu_utilization()

    # Train the model
    print("Training the model...")
    start_time = time.time()
    gbm = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data], valid_names=['eval'])
    end_time = time.time()

    # Log GPU utilization after training
    print("After training:")
    log_gpu_utilization()

    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Make predictions
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # Evaluate the model
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")



def check_tensorflow_gpu():
    print("TensorFlow GPU Check:")
    # Check if TensorFlow can detect a GPU
    if tf.config.list_physical_devices('GPU'):
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        gpu_model = tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0])['device_name']
        print(f"  GPUs detected: {num_gpus}")
        print(f"  GPU Model: {gpu_model}")
        # Perform a simple computation to confirm GPU is working
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = a + b
        print(f"  TensorFlow computation on GPU: {c.numpy()}")
    else:
        print("  No GPU detected.")

def check_pytorch_gpu():
    print("PyTorch GPU Check:")
    # Check if PyTorch can detect a GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_model = torch.cuda.get_device_name(0)
        print(f"  GPUs detected: {num_gpus}")
        print(f"  GPU Model: {gpu_model}")
        # Perform a simple computation to confirm GPU is working
        a = torch.tensor([1.0, 2.0, 3.0]).cuda()
        b = torch.tensor([4.0, 5.0, 6.0]).cuda()
        c = a + b
        print(f"  PyTorch computation on GPU: {c}")
    else:
        print("  No GPU detected.")

def check_cuda_version():
    print("CUDA Version Check:")
    try:
        cuda_version = torch.version.cuda
        print(f"  CUDA Version: {cuda_version}")
    except Exception as e:
        print(f"  Error checking CUDA version: {e}")

def check_versions():
    print("Library Versions:")
    try:
        tf_version = tf.__version__
        torch_version = torch.__version__
        print(f"  TensorFlow Version: {tf_version}")
        print(f"  PyTorch Version: {torch_version}")
    except Exception as e:
        print(f"  Error checking library versions: {e}")

def main():
    print("GPU Check for TensorFlow and PyTorch\n")
    
    print("\n\n\n")
    check_versions()
    print("\n")
    check_cuda_version()
    print("\n")
    check_tensorflow_gpu()
    print("\n")
    check_pytorch_gpu()
    print("\n")
    train_lightgbm_gpu()
    print("\n")
    check_gpu_support()


if __name__ == "__main__":
    main()

