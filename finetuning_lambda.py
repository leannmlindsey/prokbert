import yaml
import pathlib
from os.path import join
import os
import sys
import argparse
import re

from transformers import MegatronBertForMaskedLM
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from transformers import MegatronBertModel, MegatronBertConfig, MegatronBertForMaskedLM

import pkg_resources
import random
import numpy as np
import torch

from os.path import join
from prokbert.sequtils import *
from prokbert.config_utils import SeqConfig
from prokbert.training_utils import get_training_tokenizer, get_data_collator_for_overlapping_sequences
from prokbert.prok_datasets import ProkBERTPretrainingHDFDataset
# Creating the model from scratch
from prokbert.config_utils import ProkBERTConfig, get_user_provided_args
from prokbert.training_utils import *
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from prokbert.training_utils import get_default_pretrained_model_parameters, get_torch_data_from_segmentdb_classification
from prokbert.models import BertForBinaryClassificationWithPooling
from prokbert.prok_datasets import ProkBERTTrainingDatasetPT
from prokbert.config_utils import ProkBERTConfig
from prokbert.training_utils import compute_metrics_eval_prediction



seed=851115

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


def prepare_input_arguments():
    """
    Prepare and validate input arguments for ProkBERT pretraining.

    Parses command-line arguments and sets the configuration for the pretraining process.

    Returns:
        ProkBERTConfig: Configuration object for ProkBERT pretraining.
    """
    prokbert_config = ProkBERTConfig()
    keyset = ['finetuning', 'model', 'dataset', 'pretraining']
    parser, cmd_argument2group_param, group2param2cmdarg = prokbert_config.get_cmd_arg_parser(keyset)    
    args = parser.parse_args()
    user_provided_args = get_user_provided_args(args, parser)
    input_args2check = list(set(user_provided_args.keys()) - {'help'})
    parameter_group_names = list(prokbert_config.parameters.keys())
    # Initialization of the input parameterset
    parameters = {k: {} for k in parameter_group_names}
    for provided_input_argument in input_args2check:
        #print(f'Setting: {provided_input_argument}')
        param_group, param_name = cmd_argument2group_param[provided_input_argument]
        #print(f'It belongs to group: {param_group}. Maps to the parameter: {param_name}')
        act_value = getattr(args, provided_input_argument)
        parameters[param_group][param_name]=act_value
    
    #print(parameters.keys())
    
        
    _ = prokbert_config.get_and_set_model_parameters(parameters['model'])
    _ = prokbert_config.get_and_set_dataset_parameters(parameters['dataset'])
    _ = prokbert_config.get_and_set_pretraining_parameters(parameters['pretraining'])
    _ = prokbert_config.get_and_set_tokenization_parameters(parameters['tokenization'])
    _ = prokbert_config.get_and_set_segmentation_parameters(parameters['segmentation'])
    _ = prokbert_config.get_and_set_computation_params(parameters['computation'])
    _ = prokbert_config.get_and_set_datacollator_parameters(parameters['data_collator'])
    _ = prokbert_config.get_and_set_finetuning_parameters(parameters['finetuning'])

    prokbert_config.default_torchtype = torch.long
    #print(user_provided_args)

    return prokbert_config

def prepare_lambda_dataframe(dataset_split, max_length=2048):
    """
    Convert lambda dataset split to the format expected by ProkBERT.
    
    The lambda dataset has columns: 'sequence' and 'label'
    ProkBERT expects: 'segment', 'segment_id', 'y', and 'label'
    
    Args:
        dataset_split: HuggingFace dataset split
        max_length: Maximum sequence length (default 2048 to stay under ProkBERT's 2050 limit)
    """
    df = dataset_split.to_pandas()
    
    # Truncate sequences that are too long
    df['sequence'] = df['sequence'].apply(lambda x: x[:max_length] if len(x) > max_length else x)
    
    # Rename 'sequence' to 'segment'
    df['segment'] = df['sequence']
    
    # Create segment_id as a unique identifier
    df['segment_id'] = [f"seq_{i}" for i in range(len(df))]
    
    # Create 'y' column (same as label for binary classification)
    df['y'] = df['label']
    
    # Keep 'label' column as it's expected by get_torch_data_from_segmentdb_classification
    # The 'label' column is already present in the lambda dataset
    
    # Print truncation statistics
    original_lengths = dataset_split.to_pandas()['sequence'].apply(len)
    truncated_count = (original_lengths > max_length).sum()
    if truncated_count > 0:
        print(f"Truncated {truncated_count} sequences from max length {original_lengths.max()} to {max_length}")
    
    return df

def main(prokbert_config):
    """
    Main function to run the ProkBERT pretraining pipeline.

    Initializes tokenizer, data collator, dataset, and model, and then starts the pretraining process.

    Args:
        prokbert_config (ProkBERTConfig): Configuration object containing all necessary parameters for pretraining.
    """
    check_nvidia_gpu()
    print(prokbert_config.finetuning_params)


    model_name_path = prokbert_config.model_params['model_name']
    print(model_name_path)
    pretrained_model, tokenizer = get_default_pretrained_model_parameters(
    model_name=model_name_path, 
    model_class='MegatronBertModel', 
    output_hidden_states=False, 
    output_attentions=False,
    move_to_gpu=False
    )
    fine_tuned_model = BertForBinaryClassificationWithPooling(pretrained_model)

    # Loading the lambda dataset from Hugging Face
    print("Loading lambda dataset from leannmlindsey/lambda...")
    dataset = load_dataset("leannmlindsey/lambda")

    # Lambda dataset has train, validation, and test splits
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]

    # Convert to pandas dataframes with the expected column structure
    print("Preparing dataframes with required column structure...")
    train_db = prepare_lambda_dataframe(train_set)
    val_db = prepare_lambda_dataframe(validation_set)
    test_db = prepare_lambda_dataframe(test_set)

    # Display dataset info
    print(f"Train set size: {len(train_db)}")
    print(f"Validation set size: {len(val_db)}")
    print(f"Test set size: {len(test_db)}")
    print(f"Columns in train_db: {train_db.columns.tolist()}")
    print(f"First few rows of train_db:\n{train_db.head()}")

    ## Creating datasets!
    print(f'Processing train database!')
    [X_train, y_train, torchdb_train] = get_torch_data_from_segmentdb_classification(tokenizer, train_db)
    print(f'Processing validation database!')
    [X_val, y_val, torchdb_val] = get_torch_data_from_segmentdb_classification(tokenizer, val_db)
    print(f'Processing test database!')
    [X_test, y_test, torchdb_test] = get_torch_data_from_segmentdb_classification(tokenizer, test_db)
    
    train_ds = ProkBERTTrainingDatasetPT(X_train, y_train, AddAttentionMask=True)
    val_ds = ProkBERTTrainingDatasetPT(X_val, y_val, AddAttentionMask=True)
    test_ds = ProkBERTTrainingDatasetPT(X_test, y_test, AddAttentionMask=True)

    final_model_output = join(prokbert_config.model_params['model_outputpath'], prokbert_config.model_params['model_name'])

    training_args = TrainingArguments(**prokbert_config.pretraining_params)
    trainer = Trainer(
                    model=fine_tuned_model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset = val_ds,
                    compute_metrics=compute_metrics_eval_prediction,
                )
    trainer.train()
    # Saving the final model
    print(f'Saving the model to: {final_model_output}')
    fine_tuned_model.save_pretrained(final_model_output)
    
    # Optional: Evaluate on test set after training
    print(f'\nEvaluating on test set...')
    test_results = trainer.evaluate(test_ds)
    print(f'Test set results: {test_results}')


    #print(input_args)

if __name__ == "__main__":
    print(f'Parsing')

    prokbert_config = prepare_input_arguments()
    main(prokbert_config)
