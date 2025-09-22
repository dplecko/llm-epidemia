# Experimental Fine-tuning Branch

This branch contains experimental fine-tuning scripts for the LLM Epidemiology project, specifically focused on training Llama-2-7b with LoRA on NSDUH persona-based datasets.

## 🚀 Key Features

- **Custom Data Collator**: Fixed data collator implementation to handle tokenized data properly
- **Persona-based Dataset**: Rich, diverse persona generation using NSDUH data
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **Multi-GPU Support**: Automatic GPU selection and utilization
- **Wandb Integration**: Comprehensive logging and monitoring
- **Memory Optimization**: Efficient memory usage with proper caching

## 📁 Key Files

### Training Scripts
- `train_final_working.py` - **Main working training script** (recommended)
- `train_fixed_collator.py` - Fixed data collator implementation
- `train_working.py` - Alternative training approach
- `train_epidemiology_model.py` - Original training script
- `train_epidemiology_optimized.py` - Optimized version with memory management

### Dataset Generation
- `create_nsduh_persona_dataset.py` - Generates persona-based dataset from NSDUH data
- `debug_data_format.py` - Debug script for data format issues
- `test_collator.py` - Test script for data collator functionality

### Debugging & Testing
- `debug_training.py` - Comprehensive training debugging script
- `test_gpu.py` - GPU availability testing

## 🛠️ Setup Instructions

### 1. Environment Setup
```bash
# Activate conda environment
conda activate llm-epidemiology

# Install additional requirements if needed
pip install -r requirements.txt
```

### 2. Dataset Preparation
```bash
# Generate the persona-based dataset
python3 create_nsduh_persona_dataset.py
```

### 3. Training Execution
```bash
# Run the main training script
python3 train_final_working.py

# Or run in background with logging
nohup python3 train_final_working.py > training.log 2>&1 &
```

## 🔧 Technical Details

### Data Collator Fix
The main issue resolved in this branch was a data collator compatibility problem. The solution involved:

1. **Custom Data Collator**: `FixedDataCollator` class that properly handles tokenized data
2. **Proper Tokenization**: Individual text tokenization to avoid batching issues
3. **Tensor Formatting**: Correct padding and tensor creation for training

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### Memory Management
- All operations use `/local/eb/shreyas` directory for caching
- Efficient GPU memory usage (~3.5GB per GPU)
- Automatic cleanup and resource management

## 📊 Training Results

### Dataset Statistics
- **Total Examples**: 27,550
- **Training Split**: 22,040 (80%)
- **Evaluation Split**: 5,510 (20%)
- **Average Length**: ~512 tokens per example

### Model Configuration
- **Base Model**: meta-llama/Llama-2-7b-hf
- **Trainable Parameters**: 39,976,960 (0.59% of total)
- **Training Epochs**: 2
- **Batch Size**: 2 per device with 8 gradient accumulation steps

### GPU Utilization
- **Primary GPU**: 99% utilization
- **Model Sharding**: Across multiple GPUs
- **Memory Usage**: ~3.5GB per GPU
- **Training Time**: ~2-3 hours estimated

## 🔍 Monitoring

### Wandb Integration
- **Project**: llama-epidemiology
- **Real-time Metrics**: Training loss, evaluation loss, learning rate
- **Model Checkpoints**: Automatic saving and versioning

### Log Files
- `training_final_working.log` - Main training logs
- `training_fixed_collator.log` - Alternative approach logs
- GPU monitoring via `nvidia-smi`

## 🚨 Troubleshooting

### Common Issues
1. **Data Collator Errors**: Use `FixedDataCollator` class
2. **Memory Issues**: Ensure `/local/eb/shreyas` has sufficient space
3. **GPU Utilization**: Check `nvidia-smi` for GPU status
4. **Tokenization Issues**: Use individual text tokenization approach

### Debug Scripts
- `debug_training.py` - Step-by-step training debugging
- `test_collator.py` - Data collator functionality testing
- `debug_data_format.py` - Data format validation

## 📈 Performance Metrics

### Training Progress
- **Steps per Epoch**: 1,378
- **Total Steps**: 2,756
- **Learning Rate**: 2e-4 with cosine scheduling
- **Warmup Ratio**: 0.1

### Expected Outcomes
- **Training Loss**: Decreasing over epochs
- **Evaluation Loss**: Validation of model performance
- **Perplexity**: Model quality metric
- **Checkpoints**: Saved every 200 steps

## 🔄 Next Steps

1. **Monitor Training**: Check wandb dashboard for progress
2. **Evaluate Model**: Test trained model on validation set
3. **Fine-tune Parameters**: Adjust hyperparameters if needed
4. **Deploy Model**: Use trained model for inference

## 📝 Notes

- This branch contains experimental code that may have breaking changes
- Always test scripts before running on production data
- Monitor GPU usage and memory consumption
- Keep backups of important checkpoints

## 🤝 Contributing

When contributing to this branch:
1. Test all changes thoroughly
2. Update documentation as needed
3. Ensure compatibility with existing scripts
4. Follow the established code structure

---

**Status**: ✅ Training Active - Model is currently being fine-tuned on NSDUH persona dataset
**Last Updated**: September 21, 2025
**Branch**: experimental-ft
