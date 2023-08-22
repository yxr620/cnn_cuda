import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np
import time

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device_name = 'cpu'
batch_size = 512
print(device_name)


# random data for train and validation
train_data = torch.randn(1000000, 84).numpy()
train_labels = torch.randn(1000000, 1).numpy()
val_data = torch.randn(2000, 84).numpy()
val_labels = torch.randn(2000, 1).numpy()


# gpu model
model = TabNetRegressor(
    n_d=17,
    n_a=17,
    n_steps=1,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    seed=0,
    device_name=device_name
)

# train_data = train_data.to(device_name)
# train_labels = train_labels.to(device_name)
# val_data = val_data.to(device_name)
# val_labels = val_labels.to(device_name)


# train model
start = time.time()
model.fit(
    X_train=train_data,
    y_train=train_labels,
    eval_set=[(val_data, val_labels)],
    max_epochs=10,
    batch_size=batch_size
)
print("finish training")
end = time.time()

print(f"training on {device_name}, batch {batch_size}: {end - start} seconds")