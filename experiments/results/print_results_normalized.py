import os
import pickle
import numpy as np

import json
import re


def extract_first_inner_word(file_path):
    match = re.search(r'_(\w+?)_', file_path)  
    return match.group(1) if match else None




def load_and_format_conditional_volume_one_file(file_path):
    pkl_file = os.path.basename(file_path)
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
            if isinstance(results, dict):
                mean_results = {}
                std_results = {}
                for key in results[0].keys():
                    # Extract values for the current key across experiments
                    
                    values = [results[exp][key]for exp in results]
                    
                    # Remove the minimum and maximum values
                    values_sorted = sorted(values)
                    values_trimmed = values_sorted[1:-1]  # Exclude first (min) and last (max)

                    # Calculate the mean of the remaining values
                    mean_results[key] = np.mean(values_trimmed)
                    std_results[key] = np.std(values_trimmed, ddof=1)  # Sample std deviation
                # print(len(values) )
                # print(file_path)
                return mean_results, std_results
            else:
                print(f"Warning: {pkl_file} does not contain a dictionary.")
                return None, None
    except Exception as e:
        print(f"Failed to load {pkl_file}: {e}")
        return None, None
    


def load_and_format_conditional_volume_results(folder_path, alpha, n_numbers=3):
    """
    Load and format the contents of all .pkl files in the specified folder.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl') and 'tab' not in f]

    if not pkl_files:
        print("No .pkl files found in the folder.")
        return
    
    all_results = {}
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        mean_results, std_results = load_and_format_conditional_volume_one_file(file_path)
        if mean_results is not None:
            all_results[pkl_file] = (mean_results, std_results)
    
    # Generate LaTeX formatted output
    datasets = list(all_results.keys())
    keys = list(next(iter(all_results.values()))[0].keys())  # Get volume-related keys
    keys = [key for key in keys if 'ot' not in key]
    # keys = ["volume_bayes_levelsets","volume_bayes_likelihood","volume_no_bayes_levelsets","volume_no_bayes_likelihood"]

    latex_output = "Dataset " + " & " + " & ".join(keys) + "\\\\ \\hline\n"
    for dataset in sorted(datasets):  # Sorting datasets alphabetically
        if f"alpha_{alpha}" in dataset.lower():
            row = f"{dataset[:10].replace('_', ' ')} "  # Display only the first 10 characters of dataset with underscores replaced by spaces
            row = extract_first_inner_word(dataset)
            mean_results, std_results = all_results[dataset]
            for key in keys:
                # Find the smallest value in the row for the current dataset
                values = [mean_results[k] for k in keys]
                min_value = min(values)
                
                # Format the result, applying \textbf to the smallest value
                if mean_results[key] == min_value:
                    formatted_result = f"\\textbf{{{mean_results[key]:.{n_numbers}f}}} \pm {std_results[key]:.{n_numbers}f}"
                else:
                    formatted_result = f"{mean_results[key]:.{n_numbers}f} \pm {std_results[key]:.{n_numbers}f}"
                
                row += f" & ${formatted_result}$ "
            row += "\\\\ \\hline\n"
            latex_output += row

    print(latex_output)

def load_and_format_synthetic_volume_results(folder_path, alpha, n_numbers=3):
    """
    Load and format the contents of all .pkl files in the specified folder.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl') and 'tab' not in f]

    if not pkl_files:
        print("No .pkl files found in the folder.")
        return
    
    all_results = {}
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        mean_results, std_results = load_and_format_conditional_volume_one_file(file_path)
        if mean_results is not None:
            all_results[pkl_file] = (mean_results, std_results)
    
    # Generate LaTeX formatted output
    datasets = list(all_results.keys())
    keys = list(next(iter(all_results.values()))[0].keys())  # Get volume-related keys
    keys = [key for key in keys if 'ot' not in key]
    # keys = ["volume_bayes_levelsets","volume_bayes_likelihood","volume_no_bayes_levelsets","volume_no_bayes_likelihood"]

    latex_output = "Dataset " + " & " + " & ".join(keys) + "\\\\ \\hline\n"
    for dataset in sorted(datasets):  # Sorting datasets alphabetically
        if f"alpha_{alpha}" in dataset.lower():
            row = f"{dataset[:10].replace('_', ' ')} "  # Display only the first 10 characters of dataset with underscores replaced by spaces
            
            mean_results, std_results = all_results[dataset]
            for key in keys:
                # Find the smallest value in the row for the current dataset
                values = [mean_results[k] for k in keys]
                min_value = min(values)
                
                # Format the result, applying \textbf to the smallest value
                if mean_results[key] == min_value:
                    formatted_result = f"\\textbf{{{mean_results[key]:.{n_numbers}f}}} \pm {std_results[key]:.{n_numbers}f}"
                else:
                    formatted_result = f"{mean_results[key]:.{n_numbers}f} \pm {std_results[key]:.{n_numbers}f}"
                
                row += f" & ${formatted_result}$ "
            row += "\\\\ \\hline\n"
            latex_output += row

    print(latex_output)


def load_and_format_coverage_one_file(file_path):
    pkl_file = os.path.basename(file_path)
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
            if isinstance(results, dict):
                mean_results = {}
                std_results = {}
                for key in results[0].keys():
                    
                    values = [100*results[exp][key] for exp in results]

                    # Remove the minimum and maximum values
                    values_sorted = sorted(values)
                    values_trimmed = values_sorted[1:-1]  # Exclude first (min) and last (max)

                    # Calculate the mean of the remaining values
                    mean_results[key] = np.mean(values_trimmed)
                    std_results[key] = np.std(values_trimmed, ddof=1)  # Sample std deviation
            
                return mean_results, std_results
            else:
                print(f"Warning: {pkl_file} does not contain a dictionary.")
                return None, None
    except Exception as e:
        print(f"Failed to load {pkl_file}: {e}")
        return None, None





def load_and_format_coverage_results(folder_path, alpha, n_numbers=3, n_numbers_std=1):
    """
    Load and format the contents of all .pkl files in the specified folder.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl') and 'tab' not in f]

    if not pkl_files:
        print("No .pkl files found in the folder.")
        return
    
    all_results = {}
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        mean_results, std_results = load_and_format_coverage_one_file(file_path)
        if mean_results is not None:
            all_results[pkl_file] = (mean_results, std_results)
    
    # Generate LaTeX formatted output
    datasets = list(all_results.keys())
    keys = list(next(iter(all_results.values()))[0].keys())  # Get volume-related keys
    
    # remove the key if 'ot' is in the key
    keys = [key for key in keys if 'ot' not in key]
    
#     keys = ["coverage_bayes_levelsets",
# "coverage_bayes_likelihood",
# "coverage_no_bayes_levelsets",
# "coverage_no_bayes_likelihood"]

    latex_output = "Dataset " + " & " + " & ".join(keys) + "\\\\ \\hline\n"
    for dataset in sorted(datasets):  # Sorting datasets alphabetically
        if f"alpha_{alpha}" in dataset.lower():
            row = f"{dataset[:10].replace('_', ' ')} "  # Display only the first 10 characters of dataset with underscores replaced by spaces
            row = extract_first_inner_word(dataset)
            mean_results, std_results = all_results[dataset]
            for key in keys:
                # Find the smallest value in the row for the current dataset
                values = [mean_results[k] for k in keys]

                formatted_result = f"{mean_results[key]:.{n_numbers}f} \pm {std_results[key]:.{n_numbers_std}f}"
                
                row += f" & ${formatted_result}$ "
            row += "\\\\ \\hline\n"
            latex_output += row

    print(latex_output)


def load_hist(file_path, alpha):
    pkl_file = os.path.basename(file_path)
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
            if isinstance(results, dict):
                hist = {}
                for key in results[0].keys():
                    hist_values = None
                    for exp in results:
                        if hist_values is None:
                            hist_values = results[exp][key]
                        else:
                            hist_values = np.concatenate((hist_values, results[exp][key]), axis=0)
                    hist[key] = hist_values
                return hist
            else:
                print(f"Warning: {pkl_file} does not contain a dictionary.")
                return None, None
    except Exception as e:
        print(f"Failed to load {pkl_file}: {e}")
        return None, None


def load_results_empirical_conditional_coverage(folder_path, alpha):
    """
    Load and format the contents of all .pkl files in the specified folder.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl') and 'tab' not in f]

    if not pkl_files:
        print("No .pkl files found in the folder.")
        return
    
    all_results = {}
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        all_results[pkl_file] = load_hist(file_path, alpha)
    
    return all_results
