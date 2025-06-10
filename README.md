# A Diffusion-TGAN Framework for Spatio-temporal Speed Imputation — Official PyTorch implementation
Abstract— Generative Adversarial Networks (GAN) have been widely used in traffic data imputation to improve the accuracy of data imputation. However, existing GAN-based models often suffer from mode collapse and cannot fully reflect the complex characteristics of real-world traffic, which affects the quality of data imputation. To address these challenges, we incorporate the Diffusion Model (DM) into the GAN framework, integrating the traffic dynamics modeling process within the Diffusion-GAN network. Based on this, we propose a Diffusion-TGAN speed data imputation model to generate individual vehicle speeds. Combined with the generated vehicle speed, the group trajectory reconstruction result is further given. The model uses the forward process of DM to generate condition vectors to guide the training of GAN generator. Subsequently, the discriminator of GAN takes the traffic dynamics constraints into account during adversarial training. Traffic dynamics modeling aims to make the generated speed data consistent with the real traffic characteristics.
# ToDos
Initial code release
Data preprocessing
# File structure
main.py: Main training and evaluation script
dataprocess.py: preprocesing the data
readme.md: This documentation file
# Data prepocesing
This project uses trajectory data derived from raw traffic datasets such as NGSIM. Relevant microscopic features (e.g., individual vehicle speeds, space headways) are parsed and used to compute macroscopic traffic variables such as average speed and vehicle count. These are used as conditioning inputs for model training.
The preprocessing is implemented in the main script as follows:
···
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
···
# Tips
This repository contains an example of a Diffusion-TGAN. You can run python main.py to reproduce our experimental results. 
After the paper is accepted, we will release all the source code. The validation and test datasets can be accessed at https://pan.baidu.com/s/1eKWLMyWwsbJ9sRmVCluY1g. The extraction code is: jydr.
