import torch

# 1. Load your full checkpoint
checkpoint = torch.load('/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4.1.1SiLU_continue_continue/secondround_checkpoint_epoch907.pt', map_location='cpu')

# 2. Extract only the model��s weights
model_weights = checkpoint['model_state_dict']

# 3. Save those weights to a new file
torch.save(model_weights, 'model_weights_only_907.pt')
