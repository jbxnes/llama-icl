import os
import json
from typing import List, Tuple
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from words import words


# TASK SETUP
def corrupt_word(word: str, corruption_rate: float=0.7) -> str:
    """Insert random symbols into a word at a given corruption rate.
    
    Arguements:
        word (str): The word to corrupt.
        corruption_rate (int): The probability of inserting a symbol.
    """
    symbols = ['@', '#', '$', '%', '&', '*', '^', '!', '?', '+', '-', '=']
    corrupted = []
    
    for char in word:
        if random.random() < corruption_rate:
            corrupted.append(random.choice(symbols))
            corrupted.append(char)
        else:
            corrupted.append(char)
            
    return ''.join(corrupted)

def generate_corrupted_words(num_examples: List[str], corruption_rate: float=0.7) -> List[Tuple[str]]:
    """Generate (input_word, target_word) pairs.
    
    Arguements:
        num_examples (int): The number of examples to generate.
        corruption_rate (int): The probability of inserting a symbol.
    """
    examples = []
    
    for _ in range(num_examples):
        word = random.choice(words)
        examples.append((corrupt_word(word, corruption_rate), word))
        
    return examples


# MODEL EVALUATION
def evaluate_model(model_path: str, test_examples: List[Tuple[str]], include_prompt: bool=True, 
                   corruption_rate: float=0.7) -> dict:
    """Evaluate a model on the word corruption task.
    
    Arguements:
        model_path (str): The path to the model.
        test_examples (List[Tuple[str]]): A list of (input_word, target_word) pairs.
        include_prompt (bool): Whether to include the task description in the input.
        corruption_rate (int): The probability of inserting a symbol.
    """    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    results = {
        'accuracies': [],
        'predictions': {k: [] for k in [0, 1, 2, 5, 10]}
    }
    
    for k in [0, 1, 2, 5, 10]:  # Number of in-context examples
        correct = 0
        task_description = "Identify the corrupted word in the list by removing all symbols:\n\n" \
            if include_prompt else ""
            
        for corrupted_word, target_word in test_examples:
            # Build prompt
            prompt = task_description
            if k > 0:
                examples = generate_corrupted_words(k, corruption_rate)
                for ex_in, ex_out in examples:
                    prompt += f"Input: {ex_in}\nOutput: {ex_out}\n\n"
            
            prompt += f"Input: {corrupted_word}\nOutput:"
            
            # Generate prediction
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=12,
                pad_token_id=tokenizer.eos_token_id
            )
            prediction = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):]).strip()
            
            is_correct = prediction.startswith(target_word)
            results['predictions'][k].append({
                'input_word': corrupted_word,
                'target_word': target_word,
                'prediction': prediction,
                'correct': is_correct,
                'prompt': prompt
            })
            
            # Check correctness
            if is_correct:
                correct += 1
        
        accuracy = 100 * correct / len(test_examples)
        results['accuracies'].append(accuracy)
        print(f"K={k}: {accuracy:.1f}%")
    
    return results


# RUN EXPERIMENT
def run(model_list: List[str], root_dir: str, corruption_rate: float=0.7, run_id: str=None):
    """Evaluate a list of models on the word corruption task and save the results.
    
    Arguements:
        model_list (List[str]): A list of model names.
        root_dir (str): The root directory for the model weights.
        corruption_rate (int): The probability of inserting a symbol.
        run_id (optional): A unique identifier for the run.
    """
    test_examples = [(corrupt_word(word, corruption_rate), word) for word in words]
    
    for include_prompt in [True, False]:
        results = {}
        
        for model_name in model_list:
            print(f"\nEvaluating {model_name}...")
            
            model_path = root_dir + model_name
            results[model_name] = evaluate_model(model_path, 
                                                 test_examples,
                                                 include_prompt=include_prompt, 
                                                 corruption_rate=corruption_rate)
        # Save the results to a JSON file
        os.makedirs('results', exist_ok=True)
        
        if run_id is not None:
            filename = f"./results/{run_id}_results_prompt={include_prompt}.json"
        else:
            filename = f"./results/results_prompt={include_prompt}.json"
            
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=str, default=['Llama-3.2-1B', 'Llama-3.2-3B', 'Meta-Llama-3.1-8B'])
    parser.add_argument("--root_dir", type=str, default='./')
    parser.add_argument("--corruption_rate", type=float, default=0.7)
    parser.add_argument("--run_id", type=str, default='')
    
    args = parser.parse_args()
    
    run(args.models, args.root_dir, args.corruption_rate, args.run_id)
