# MDSTNet
This repository contains the official implementation of the paper: "Air Quality Prediction with A Meteorology-Guided Modality-Decoupled Spatio-Temporal Network"

## Usage

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c "./config/MDSTNet.json"
```

### Testing

```bash 
CUDA_VISIBLE_DEVICES=0 python st_test.py -r "PATH_TO_CHECKPOINT"
```

## Model Configuration

The model configurations can be found in `config/MDSTNet.json`. Key hyperparameters include:

- seq_len: Input sequence length
- pred_len: Prediction sequence length  
- d_model: Dimension of model
- n_heads: Number of attention heads
- e_layers: Number of encoder layers
- d_layers: Number of decoder layers

## Project Structure

```
MDSTNet/
├── base/               # Base classes for data loader, model and trainer
├── config/            # Configuration files
├── criterion/         # Loss functions
├── data_loader/       # Data loading utilities
├── evaluation/        # Evaluation metrics
├── layers/            # Model layers implementation  
├── logger/            # Logging utilities
├── model/            # MDSTNet model implementation
├── trainer/           # Training logic
└── utils/            # Utility functions
```

## Requirements

```bash
torch==1.12.1+cu102 
torchvision==0.13.1+cu102 
torchaudio==0.12.1
numpy
pandas
tqdm
```

## Citation

If you find this code useful for your research, please cite our paper:

```
@article{MDSTNet,
  title={Air Quality Prediction with A Meteorology-Guided Modality-Decoupled Spatio-Temporal Network},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
