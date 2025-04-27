import os
import re
import glob
import json
import random

# Input and output directories
input_dir = r"C:\Users\Gattupalli Saketh\OneDrive\Desktop\Dataset"
output_dir = r"C:\Users\Gattupalli Saketh\OneDrive\Desktop\Preprocessed"
output_json = os.path.join(output_dir, "cuda_dataset.json")

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Supporting file extensions (standardized)
valid_extensions = ['.cu', '.cuh', '.cpp', '.h', '.hpp', '.hxx']

# Regular expressions for non-code elements
comment_pattern = r'/\*[\s\S]*?\*/|//.*$'  # Fixed typo in comment pattern
license_pattern = r'^\s*(/\*[\s\S]*?(?:LICENSE|COPYRIGHT)[\s\S]*?\*/|//.*?(?:LICENSE|COPYRIGHT).*?\n)'  # Target license headers
pragma_once_pattern = r'^\s*#pragma\s+once\s*$'  # Matches #pragma once
ide_config_pattern = r'^\s*#\s*(ifdef|ifndef|endif)\s+_(DEBUG|WIN32|_MSC_VER).*?$'  # Specific IDE directives

def clean_code(content):
    """Remove non-code elements from file content."""
    # Remove multi-line and single-line comments
    content = re.sub(comment_pattern, '', content, flags=re.MULTILINE)
    # Remove license headers
    content = re.sub(license_pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
    # Remove #pragma once
    content = re.sub(pragma_once_pattern, '', content, flags=re.MULTILINE)
    # Remove IDE-specific configurations
    content = re.sub(ide_config_pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
    # Remove multiple empty lines and trim whitespace
    content = re.sub(r'\n\s*\n+', '\n', content).strip()
    return content

def create_fim_example(code):
    """Create FIM (Fill-in-the-Middle) example for LLM training."""
    if len(code) < 50:  # Skip short snippets
        return None
    # Randomly split code for FIM
    split_point = random.randint(int(0.2 * len(code)), int(0.8 * len(code)))
    prefix = code[:split_point]
    middle = code[split_point:split_point + min(200, len(code) - split_point)]
    suffix = code[split_point + len(middle):]
    fim_text = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}"
    return {
        "prefix": prefix,
        "middle": middle,
        "suffix": suffix,
        "fim_text": fim_text
    }

def process_file(filepath, output_filepath, json_data):
    """Read, clean, and save a single file; add to JSON data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the content
        cleaned_content = clean_code(content)
        
        # Save only if there's meaningful content left
        if cleaned_content.strip():
            # Save to individual file
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Processed: {filepath} -> {output_filepath}")
            
            # Create JSON entry
            rel_path = os.path.relpath(filepath, input_dir)
            json_entry = {
                "file_path": rel_path,
                "cleaned_code": cleaned_content
            }
            # Add FIM fields (optional, for CodeGemma training)
            fim_data = create_fim_example(cleaned_content)
            if fim_data:
                json_entry.update(fim_data)
            
            json_data.append(json_entry)
        else:
            print(f"Skipped: {filepath} (empty after cleaning)")
    except UnicodeDecodeError:
        print(f"Skipped: {filepath} (encoding error)")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # Collect data for JSON
    json_data = []
    
    # Find all files with valid extensions
    for ext in valid_extensions:
        for filepath in glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True):
            # Construct output path
            rel_path = os.path.relpath(filepath, input_dir)
            output_filepath = os.path.join(output_dir, rel_path)
            
            # Process the file and add to JSON data
            process_file(filepath, output_filepath, json_data)
    
    # Save JSON data
    if json_data:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(json_data)} entries to {output_json}")
    else:
        print("No data to save to JSON")

if __name__ == "__main__":
    main()


