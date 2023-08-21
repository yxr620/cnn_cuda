import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device_name = "cpu"
print(device_name)
# 创建特征和标签的示例数据
X_train = np.array(torch.randn((100, 84)))
y_train = np.array(torch.randn((100, 1)))

# 定义TabNetRegressor模型
# implement the following model using pytorch_tabnet, and place the model on gpu
# StackedTabNetRegressor(feature_columns=None,
#                        num_regressors=1,
#                        num_layers=2,
#                        num_features=84,
#                        feature_dim=17,
#                        output_dim=16,
#                        num_decision_steps=1)


# gpu model
model = TabNetRegressor(n_d=17, n_a=17, n_steps=1, gamma=1.5, n_independent=2, n_shared=2, seed=0, device_name=device_name)

# 训练模型
model.fit(X_train=X_train, y_train=y_train, max_epochs=100)

# 使用模型进行预测
X_test = torch.randn((10, 84))
predictions = model.predict(X_test)

print(predictions)