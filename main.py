import os
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import random
from thop import profile
import time
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

def compute_mape(y_true, y_pred):
    # Convert to numpy if tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_flops_and_inference_time(model, input_dim):
    dummy_input = torch.randn(1, input_dim * 2)  
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2 

    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy_input)
    avg_inference_time = (time.time() - start_time) / 100

    return flops, params, avg_inference_time

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     
setup_seed(6)

def plot_speed_distribution(input_data, output_data, idx, vehicle_count_max, subset_data):
    input_at_t = input_data[idx]
    output_at_t = output_data[idx]
    individual_speeds_at_t = input_at_t[:vehicle_count_max] 
    real_speeds = subset_data['Individual_Speeds'].iloc[idx] 
    
    if len(real_speeds) < vehicle_count_max:
            real_speeds = np.pad(real_speeds, (0, vehicle_count_max - len(real_speeds)), mode='constant', constant_values=0)
    else:
            real_speeds = real_speeds[:vehicle_count_max]
    plt.figure(figsize=(10, 6))
    plt.plot(range(vehicle_count_max), denormalize(individual_speeds_at_t), label="Real Individual Speed")  # 使用实际车速
    plt.plot(range(vehicle_count_max), denormalize(output_at_t), label="Predicted Speed")  # 使用生成车速
    plt.plot(range(vehicle_count_max), real_speeds, label="Original Real Speed", linestyle='--')  # 添加原始车速线

    plt.xlabel("Vehicle Index")
    plt.ylabel("Speed (km/h)")
    plt.title(f"Speed Distribution at Time {idx}")
    plt.legend()
    plt.show() 

def compute_traffic_penalty(predicted_speed, true_speed, headway, free_flow_speed=50, jam_density=200, critical_headway=2):
    expected_speed = normalize(traffic_model(headway, free_flow_speed, jam_density, critical_headway))
    speed_diff = torch.abs(predicted_speed - expected_speed)
    penalty = torch.mean(speed_diff)  
    return penalty

def traffic_model(headway, free_flow_speed, jam_density, critical_headway):
    if isinstance(headway, np.ndarray):
        headway = torch.tensor(headway)
    speed = torch.zeros_like(headway)  
    mask = torch.lt(headway, critical_headway) 
    speed[mask] = free_flow_speed * (1 - (1 / headway[mask]) / jam_density)
    speed[~mask] = free_flow_speed
    return speed

def generator_loss(predicted_speed, true_speed, headway, free_flow_speed, jam_density, critical_headway):
    mse_loss = compute_mse_loss(true_speed, predicted_speed)
    traffic_penalty = compute_traffic_penalty(predicted_speed, true_speed, headway, free_flow_speed, jam_density, critical_headway)
    total_loss = mse_loss*0.7+ traffic_penalty*0.3
    return total_loss

# 数据加载与处理
data = pd.read_csv('data.csv')
def denormalize(data, min_speed=0, max_speed=50):
    return (data + 1) * (max_speed - min_speed) / 2 + min_speed
def normalize(data, min_speed=0, max_speed=50):
    return 2 * (data - min_speed) / (max_speed - min_speed) - 1

def parse_speed_list(speed_list_str):
    return np.array([float(x) for x in re.findall(r'-?\d+\.?\d*', speed_list_str)])

# 解析速度数据（个体车速）
data['Individual_Speeds'] = data['Individual_Speeds'].apply(parse_speed_list)

# 解析车头时距（每个时刻所有车辆的车头时距）
data['Individual_Headways'] = data['Individual_Headways'].apply(parse_speed_list)

# 解析每个时刻的平均车速
data['Mean_Speed'] = data['Mean_Speed'].apply(lambda x: float(x))  # 确保是数值型数据

# 限制数据量以加速训练
#subset_data = data.sample(500)  # 随机抽取100行
subset_data = data[(data['Vehicle_Count'] >= 50) & (data['Vehicle_Count'] <= 99)]

print(subset_data['Vehicle_Count'].min())  # 检查最小的车辆数
print(subset_data['Vehicle_Count'].max())  # 检查最大的车辆数
all_speeds = []

for speeds in subset_data['Individual_Speeds']:
    all_speeds.extend(speeds)  # 扩展每一行的速度数据，形成一个完整的列表
# 计算方差和标准差
speeds_series = pd.Series(all_speeds)  # 转换为 Pandas Series 以便进行计算
variance = speeds_series.var()  # 方差
std_deviation = speeds_series.std()  # 标准差
print(f"方差: {variance}")
print(f"标准差: {std_deviation}")
vehicle_count_max = max(subset_data['Vehicle_Count'])
print(f"最大车辆数为：{vehicle_count_max}") 

input_data = []  # 初始化 input_data 列表
for t in range(len(subset_data)):
    headways_at_t = subset_data['Individual_Headways'].iloc[t]
    avg_speed_at_t = subset_data['Mean_Speed'].iloc[t]
    
    # 填充或截断车头时距到 vehicle_count_max 的长度
    if len(headways_at_t) < vehicle_count_max:
        headways_padded = np.pad(headways_at_t, (0, vehicle_count_max - len(headways_at_t)), mode='constant', constant_values=0)
    else:
        headways_padded = headways_at_t[:vehicle_count_max]
    
    # 填充平均车速到 vehicle_count_max 的长度
    avg_speeds_padded = [avg_speed_at_t] * vehicle_count_max  # 将均速填充到 vehicle_count_max 长度
    
    # 将车头时距与平均车速合并，作为输入数据
    input_data.append(np.concatenate([headways_padded, avg_speeds_padded], axis=0))

train_input_data = np.array(input_data)

# 生成训练集输出数据（train_set 输出数据）
train_output_data = []
for speeds in subset_data['Individual_Speeds']:
    if len(speeds) < vehicle_count_max:
        speeds_padded = np.pad(speeds, (0, vehicle_count_max - len(speeds)), mode='constant', constant_values=0)
    else:
        speeds_padded = speeds[:vehicle_count_max]
    train_output_data.append(speeds_padded)

train_output_data = np.array(train_output_data)

# 获取训练集和测试集
train_data_size = int(len(subset_data) * 0.7)  
train_set = subset_data[:train_data_size]  
test_set = subset_data[train_data_size:]  

# 定义测试集字典（test_uis_dict）
test_uis_dict = {}
for t in range(len(test_set)):
    test_uis_dict[t] = test_set['Individual_Speeds'].iloc[t].tolist()  
train_input_data_normalized = normalize(train_input_data, min_speed=0, max_speed=40)
train_output_data_normalized = normalize(train_output_data, min_speed=0, max_speed=40)
test_input_data = []
test_output_data = []
for t in range(len(test_set)):
    headways_at_t = test_set['Individual_Headways'].iloc[t]  # 获取时刻 t 的车头时距
    avg_speed_at_t = test_set['Mean_Speed'].iloc[t]  # 获取时刻 t 的平均车速
    
    # 填充或截断车头时距到 vehicle_count_max 的长度
    if len(headways_at_t) < vehicle_count_max:
        headways_padded = np.pad(headways_at_t, (0, vehicle_count_max - len(headways_at_t)), mode='constant', constant_values=0)
    else:
        headways_padded = headways_at_t[:vehicle_count_max]

    avg_speeds_padded = [avg_speed_at_t] * vehicle_count_max  # 将均速填充到 vehicle_count_max 长度
    
    test_input_data.append(np.concatenate([headways_padded, avg_speeds_padded], axis=0))

    speeds = test_set['Individual_Speeds'].iloc[t] 
    if len(speeds) < vehicle_count_max:
        speeds_padded = np.pad(speeds, (0, vehicle_count_max - len(speeds)), mode='constant', constant_values=0)
    else:
        speeds_padded = speeds[:vehicle_count_max]
    test_output_data.append(speeds_padded)

test_input_data = np.array(test_input_data)
test_output_data = np.array(test_output_data)

test_input_data_normalized = normalize(test_input_data, min_speed=0, max_speed=40)
test_output_data_normalized = normalize(test_output_data, min_speed=0, max_speed=40)


test_dataset = TensorDataset(torch.Tensor(test_input_data_normalized), torch.Tensor(test_output_data_normalized))
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)


print(f"Train input data shape: {train_input_data.shape}")
print(f"Train output data shape: {train_output_data.shape}")
print(f"Test uis dict (keys): {list(test_uis_dict.keys())[:5]}")
print(f"Test uis dict (values): {list(test_uis_dict.values())[:5]}")
z_dim=100

class Gen(nn.Module):
    def __init__(self, nb_item):
        super(Gen, self).__init__()
        self.fc1 = nn.Linear(nb_item * 2, 256)
        self.fc2 = nn.Linear(256, nb_item)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.clamp(x, min=0, max=40)  
        return x

class Dis(nn.Module):
    def __init__(self, nb_item):
        super(Dis, self).__init__()
        self.fc1 = nn.Linear(nb_item, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max, steps, device):
        super(GaussianDiffusion, self).__init__()
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

    def q_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        return x_0 + noise * self.noise_scale

    def sample_posterior(self, x_0, t):
        return x_0  

    def sample_timesteps(self, steps, batch_size):
        return torch.randint(0, steps, (batch_size,)).to(self.device)
def compute_l1_loss(y_true, y_pred):
    return torch.abs(y_true - y_pred).mean()

def compute_mse_loss(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)

    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    return ((y_true - y_pred) ** 2).mean()

def test_DGRM(test_input_data, test_output_data, trained_gen):
    trained_gen.eval()  
    predictions = []
    for i in range(len(test_input_data)):
        input_data = torch.Tensor(test_input_data[i])
        output = trained_gen(input_data)
        predictions.append(output.detach().numpy())

    # 测试结果打印
    predictions = np.array(predictions)
    mse = np.mean((predictions - test_output_data) ** 2)  # 计算 MSE
    mape = compute_mape(test_output_data, predictions)
    print("测试结果 - MSE:", mse)
    print(f"测试结果 - MAPE: {mape:.2f}%") 
    return mse, mape

def train_DGRM(save_dir, train_input_data, train_output_data, nb_item, epoches, batch_size, test_input_data, test_output_data):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model_1.pkl')
    gen = Gen(nb_item)
    dis_ra = Dis(nb_item)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=0.0002, weight_decay=1e-2, betas=(0.5, 0.999))
    dis_ra_opt = torch.optim.Adam(dis_ra.parameters(), lr=0.0002, weight_decay=1e-2, betas=(0.5, 0.999))
    mean_type = 'x0'
    noise_schedule = 'linear-var'
    noise_scale = 0.1
    noise_min = 0.1
    noise_max = 1
    steps = 5
    diffusion = GaussianDiffusion(mean_type, noise_schedule, noise_scale, noise_min, noise_max, steps + 1, device='cpu')
    dataset = torch.utils.data.TensorDataset(torch.Tensor(train_input_data), torch.Tensor(train_output_data))
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    free_flow_speed = 50 
    jam_density = 200 
    critical_headway = 2 

    # 开始训练
    for e in range(epoches):
        #------------------------------------------
        # Train D
        #------------------------------------------
        dis_ra.train()
        gen.eval()
        for step in range(4):  # 示例：步数
            for idxs, _ in dataloader:
                                idxs = idxs.long()
                idxs = torch.clamp(idxs, 0, len(train_input_data) - 1)
                headways_at_t = train_input_data[idxs, :vehicle_count_max]  # 获取车头时距
                # 使用扩散模型的正向过程生成噪声数据
                noisy_data = diffusion.q_sample(torch.Tensor(train_input_data[idxs]), t=0)  # 将数据转为噪声
                # 使用生成器进行生成
                output = gen(noisy_data)  # 从生成器获取输出
                # 计算交通动力学惩罚
                traffic_penalty = compute_traffic_penalty(output, train_output_data[idxs], headways_at_t, free_flow_speed, jam_density, critical_headway)
                # 计算总损失
                loss = generator_loss(output, train_output_data[idxs], headways_at_t, free_flow_speed, jam_density, critical_headway)
                gen_opt.zero_grad()
                loss.backward()
                gen_opt.step()

        # ------------------------------------------
        # Train G
        # ------------------------------------------
        gen.train()
        dis_ra.eval()
        for step in range(2): 
            for idxs, _ in dataloader:
                idxs = idxs.long()  
                idxs = torch.clamp(idxs, 0, len(train_input_data) - 1)
                headways_at_t = train_input_data[idxs, :vehicle_count_max]  # 获取车头时距
                # 获取输出
                noisy_data = diffusion.q_sample(torch.Tensor(train_input_data[idxs]), t=0)  # 将数据转为噪声
                output = gen(noisy_data)  # 从生成器获取输出
                # 计算交通动力学惩罚
                traffic_penalty = compute_traffic_penalty(output, train_output_data[idxs], headways_at_t, free_flow_speed, jam_density, critical_headway)
                # 计算总损失
                loss = generator_loss(output, train_output_data[idxs], headways_at_t, free_flow_speed, jam_density, critical_headway)
                gen_opt.zero_grad()
                loss.backward()
                gen_opt.step()

        # 每个 epoch 结束后，打印损失并评估
        with torch.no_grad():
            output = gen(torch.Tensor(train_input_data)) 
            mse_loss = compute_mse_loss(torch.Tensor(train_output_data), output)  # 计算均方误差
            print(f"Epoch {e + 1}, MSE Loss: {mse_loss.item()}")
            
            # 可视化某个时刻的输出
            if (e + 1) % 100 == 0: 
                plot_speed_distribution(train_output_data, output, idx=0, vehicle_count_max=nb_item, subset_data=subset_data)
            print(f"Epoch {e + 1} completed.")
            torch.save(gen.state_dict(), save_path)
            mse = test_DGRM(test_input_data, test_output_data, gen)
            print(f"测试 MSE Loss: {mse}")



if __name__ == '__main__':
    save_dir = r'XXXX'
    train_DGRM(save_dir=save_dir, 
               train_input_data=train_input_data_normalized, 
               train_output_data=train_output_data_normalized,
               nb_item=vehicle_count_max, epoches=10, batch_size=32,
               test_input_data=test_input_data, test_output_data=test_output_data)

