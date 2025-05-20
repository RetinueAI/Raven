# Raven: A Token Classification Model

## Description

Raven is a deep learning model designed for token-level classification tasks in natural language processing (NLP). It leverages the power of pre-trained BERT models combined with additional neural network layers to classify each token in a given text into one of several predefined classes. The model is particularly suited for tasks requiring fine-grained analysis of text, such as named entity recognition, part-of-speech tagging, or segment classification.

## Dataset

The model is trained on the "Raven Dataset," a custom dataset generated from multiple CSV files located in the './data' directory. The dataset is created using the `__generate_dataset.py` script, which processes these files to create paired sentences from different label categories, designed to capture transitions or continuations between different topics or styles. The resulting dataset is saved as "__raven.csv". Additionally, the script generates a `config.txt` file that specifies the number of classes (`n_classes`) used in the model.

To generate the dataset, ensure the CSV files are in the './data' directory and run:

```bash
python __generate_dataset.py
```

## Model Architecture

The Raven model architecture includes:

- **BERT (bert-base-multilingual-cased)**: Provides contextual token embeddings for multilingual text understanding.
- **Conv1d**: A 1D convolutional layer to extract local features from the BERT embeddings.
- **LSTM**: A Long Short-Term Memory layer to capture sequential dependencies in the data.
- **Transformer Encoder**: Four layers of TransformerEncoderLayer for advanced sequence modeling using self-attention mechanisms.
- **Classifier**: A linear layer to map the encoded features to the desired number of classes (default: 5).

## Training

Training is performed using the following setup:

- **Batch Size**: 4
- **Learning Rate**: 3e-7
- **Epochs**: Up to 100, with early stopping based on validation loss
- **Optimizer**: AdamW
- **Scheduler**: ReduceLROnPlateau with patience of 2
- **Loss Function**: CrossEntropyLoss

Mixed precision training is used for better performance on GPU-enabled systems. The dataset is split into 90% training and 10% testing.

To train the model, run:

```bash
python __train.py
```

## Inference

To use the model for inference:

1. Load the trained model weights from the "weights" directory.
2. Tokenize the input text using the BERT tokenizer (`bert-base-multilingual-cased`).
3. Pass the tokenized inputs through the model to obtain logits.
4. Apply softmax to get probabilities and argmax to determine the predicted class for each token.

To perform inference, run:

```bash
python __inference.py
```

Ensure the trained weights are available in the "weights" directory.

## Testing

The project includes a test script (`__test.py`) that verifies the alignment of labels with input tokens in the dataset. It checks for:

- Mismatches in the number of tokens and labels.
- Correct labeling of special tokens ([CLS], [SEP], [PAD]) with -100.
- Ensures non-special tokens have valid labels.

This ensures the dataset is correctly formatted for training and inference.

To run the tests, execute:

```bash
python __test.py
```

## Dependencies

The project requires the following Python packages:

| Package         | Version  |
|-----------------|----------|
| transformers    | 4.38.2   |
| torch           | 2.2.1    |
| pandas          | 2.2.1    |
| scikit-learn    | 1.4.1    |
| tqdm            | 4.66.2   |
| wandb           | 0.16.3   |

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the project's repository to suggest improvements or report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, contact marius.hanssen@retinueai.com or josefjameshard@retinueai.com
