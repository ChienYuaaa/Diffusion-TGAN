import pandas as pd
trajectories_data_path = 'trajectories.csv'
trajectories_data = pd.read_csv(trajectories_data_path)

def generate_corrected_dataset(trajectory_data):
    grouped = trajectory_data.groupby('Frame_ID')
    frame_ids, vehicle_counts, mean_speeds = [], [], []
    individual_speeds, positions, following_flags = [], [], []

    for frame_id, frame_data in grouped:
        frame_ids.append(frame_id)
        vehicle_counts.append(frame_data['Vehicle_ID'].nunique())
        mean_speeds.append(frame_data['v_Vel'].mean())
        individual_speeds.append(frame_data['v_Vel'].tolist())
        positions.append(frame_data[['Vehicle_ID', 'Local_X', 'Local_Y']].to_dict('records'))
        following_flags.append(frame_data[['Vehicle_ID', 'Preceding', 'Following']].to_dict('records'))

    corrected_df = pd.DataFrame({
        'Frame_ID': frame_ids,
        'Vehicle_Count': vehicle_counts,
        'Mean_Speed': mean_speeds,
        'Individual_Speeds': individual_speeds,
        'Positions': positions,
        'Following_Behavior': following_flags
    })

    return corrected_df

# 生成增强数据集
enhanced_corrected_data = generate_corrected_dataset(trajectories_data)

# 保存到文件
enhanced_corrected_data_path = 'enhanced_corrected_trajectory_data.csv'
enhanced_corrected_data.to_csv(enhanced_corrected_data_path, index=False)

print(f"文件已保存为: {enhanced_corrected_data_path}")
