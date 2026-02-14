# SQL-to-Text Training and Evaluation

Model-agnostic framework for fine-tuning and evaluating models for SQL-to-text generation using the PAUQ dataset.

## Features

- **Model-agnostic**: Automatically detects and supports both seq2seq (RuT5, T5, BART) and causal LM (Qwen, Llama, Mistral) architectures
- **Multiple metrics**: BLEU, ROUGE, CHRF evaluation
- **Easy to use**: Simple CLI interface with all common training parameters
- **Colab-ready**: Full notebook for Google Colab training

## Supported Model Types

### Seq2Seq Models (Encoder-Decoder)
- `cointegrated/rut5-base` - Russian T5 (recommended)
- `cointegrated/rut5-small` - Smaller Russian T5
- `google/flan-t5-base` - English T5
- `facebook/bart-base` - BART
- Any T5/BART variant from HuggingFace Hub

### Causal LMs (Decoder-Only)
- `Qwen/Qwen2.5-0.5B-Instruct` - Qwen
- `microsoft/Phi-3-mini-4k-instruct` - Phi-3
- `meta-llama/Llama-3.2-1B-Instruct` - Llama
- Any decoder-only model from HuggingFace Hub

## Installation

```bash
# Clone the repository
git clone git@github.com:madrugado/sql-to-text.git
cd sql-to-text

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

## Dataset

Place your PAUQ dataset files in the `data/` directory:
- `data/pauq_train.json` - Training data
- `data/pauq_dev.json` - Validation/test data

## Training

### Basic Usage

```bash
python train_sql_to_text.py --model cointegrated/rut5-base
```

### Full Parameters

```bash
python train_sql_to_text.py \
    --model cointegrated/rut5-base \
    --output-dir ./output/my_model \
    --data-dir ./data \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --num-epochs 3 \
    --learning-rate 2e-4 \
    --max-length 512 \
    --mode train
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `cointegrated/rut5-base` | HuggingFace model name or local path |
| `--output-dir` | str | `./output/sql_to_text` | Directory to save the trained model |
| `--data-dir` | str | `./data` | Directory containing PAUQ data files |
| `--batch-size` | int | 4 | Per-device batch size |
| `--gradient-accumulation-steps` | int | 4 | Gradient accumulation steps |
| `--num-epochs` | int | 3 | Number of training epochs |
| `--learning-rate` | float | 2e-4 | Learning rate |
| `--max-length` | int | 512 | Maximum sequence length |
| `--mode` | str | `both` | Mode: `train`, `eval`, or `both` |

### Modes

- `train` - Only train the model
- `eval` - Only evaluate (requires `--model-path` for evaluation)
- `both` - Train then evaluate (default)

## Evaluation

```bash
# Evaluate a trained model
python evaluate_qwen.py \
    --model-path ./output/sql_to_text \
    --data-dir ./data \
    --num-samples 100
```

### Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | str | `cointegrated/rut5-base` | Path or HuggingFace model to evaluate |
| `--data-dir` | str | `./data` | Directory containing PAUQ data files |
| `--num-samples` | int | `None` (all) | Number of samples to evaluate |
| `--output-file` | str | `None` | Path to save predictions JSON |
| `--max-new-tokens` | int | 100 | Maximum tokens to generate |
| `--temperature` | float | 0.7 | Sampling temperature |
| `--no-sample` | flag | `False` | Use greedy decoding instead of sampling |

### Evaluation Metrics

The evaluation script computes:
- **BLEU-1 to BLEU-4** - n-gram precision scores
- **ROUGE-1, ROUGE-2, ROUGE-L** - Recall-oriented metrics
- **CHRF** - Character-level F-score
- **LaBSE Similarity** - Semantic similarity using LaBSE embeddings

Results are automatically saved to `evaluation_results.csv`.

## Google Colab

For easy training without local GPU, use the provided Colab notebook:

1. Open `sql_to_text_colab.ipynb` in Google Colab
2. Run the cells sequentially
3. Upload your data files when prompted
4. Configure the model and training parameters
5. Run training and evaluation

The notebook includes:
- Dependency installation
- Google Drive mounting (optional)
- Data file upload
- GPU detection
- Model training
- Evaluation with metrics
- Model download

## Examples

### Train RuT5 (Recommended for Russian)

```bash
python train_sql_to_text.py --model cointegrated/rut5-base
```

### Train Qwen (Causal LM)

```bash
python train_sql_to_text.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### Evaluate on Full Dev Set

```bash
python evaluate_qwen.py --model-path ./output/sql_to_text
```

### Quick Evaluation on 50 Samples

```bash
python evaluate_qwen.py --model-path ./output/sql_to_text --num-samples 50
```

## Model Detection

The training script automatically detects the model type and uses the appropriate architecture:

- **Seq2Seq models**: Uses `AutoModelForSeq2SeqLM`, `Seq2SeqTrainer`, and `DataCollatorForSeq2Seq`
- **Causal LMs**: Uses `AutoModelForCausalLM`, `Trainer`, and `DataCollatorForLanguageModeling`

No need to specify model type manually!

## Data Format

The PAUQ dataset should have the following JSON structure:

```json
[
  {
    "id": "sample_1",
    "question": {
      "en": "How many singers do we have?"
    },
    "query": {
      "en": "SELECT count(*) FROM singer;"
    },
    "sql": "SELECT count(*) FROM singer;"
  },
  ...
]
```

## Training Tips

### GPU Memory Optimization

- Reduce `--batch-size` if you run out of memory
- Increase `--gradient-accumulation-steps` to maintain effective batch size
- Use smaller models for faster training on limited resources

### Recommended Settings

| GPU | Model | Batch Size | Grad Accum |
|-----|-------|------------|-------------|
| T4 (Colab) | RuT5-base | 4 | 4 |
| A100 (Colab) | RuT5-base | 16 | 2 |
| T4 (Colab) | Qwen-0.5B | 2 | 4 |
| A100 (Colab) | Qwen-0.5B | 8 | 2 |

### Training Time Estimates

- RuT5-base on T4: ~15-20 minutes/epoch
- Qwen-0.5B on T4: ~20-30 minutes/epoch
- RuT5-base on A100: ~5-10 minutes/epoch
- Qwen-0.5B on A100: ~8-15 minutes/epoch

## Output Structure

After training, the output directory contains:

```
output/sql_to_text/
├── config.json           # Model configuration
├── tokenizer_config.json # Tokenizer configuration
├── pytorch_model.bin     # Model weights (or safetensors)
├── tokenizer.json        # Tokenizer files
├── special_tokens_map.json
├── training_args.bin     # Training arguments
├── trainer_state.json    # Training state
└── checkpoint-*         # Intermediate checkpoints
```

## License

This project uses the PAUQ dataset and pre-trained models from HuggingFace Hub. Please check the respective licenses for each component.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
