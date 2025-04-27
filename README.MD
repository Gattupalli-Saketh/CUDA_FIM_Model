# 🚀 CodeGemma CUDA Fine-Tuning Project

Welcome to the **CodeGemma CUDA Fine-Tuning** repository! This project demonstrates how to fine-tune the powerful **CodeGemma** model for generating CUDA code snippets using your custom dataset. By utilizing LoRA (Low-Rank Adaptation), we can efficiently train the model without requiring massive computational resources.

💡 **Key Features**:
- Fine-tuning CodeGemma for CUDA code generation.
- Efficient LoRA-based training for faster adaptation.
- Generate CUDA code based on provided prefixes and suffixes.

## 📚 Requirements

Before you start, you'll need to install the following dependencies:

```bash
pip install transformers datasets peft trl torch
Required Libraries:
transformers: For model handling.

datasets: For loading and processing datasets.

peft: For Low-Rank Adaptation (LoRA).

torch: For PyTorch operations and GPU acceleration.

json: To handle and parse JSON data.

📁 Project Structure
Here's what you'll find in this repository:

finetuning.py: Full script for loading datasets, fine-tuning the model, and generating code.

cuda_dataset.json: Custom dataset for training (replace with your own dataset).

README.md: This file, providing an overview and instructions.

LICENSE: Project license (MIT License).

requirements.txt: List of all dependencies.

🛠️ Steps to Run
1️⃣ Upload Your Custom Dataset
Start by preparing your own JSON dataset containing CUDA code snippets. Each entry should have a key "fim_text" containing the formatted text you want to use for fine-tuning.

json
Copy
Edit
[
  {
    "fim_text": "<fim_prefix>#include <stdio.h>\n#include <cuda.h>\n\n__global__ void add(int *a, int *b, int *c) { ... }<fim_suffix>"
  },
  {
    "fim_text": "<fim_prefix>int main(void) { ... }<fim_suffix>"
  }
]
2️⃣ Load the Pre-Trained CodeGemma Model
You can load the CodeGemma model from Hugging Face directly:

python
Copy
Edit
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/codegemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
3️⃣ Preprocess the Dataset
Use the following Python code to load your JSON dataset and format it for training. The fim_text field is crucial for fine-tuning.

python
Copy
Edit
import json

def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

raw_data = load_json_dataset("cuda_dataset.json")

# Format the dataset
formatted_data = [{"text": item["fim_text"]} for item in raw_data]
4️⃣ Fine-Tune the Model with LoRA
LoRA allows us to efficiently fine-tune the model without excessive computational overhead. Here’s how to configure LoRA and fine-tune the model.

python
Copy
Edit
from peft import LoraConfig
from trl import SFTTrainer

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    peft_config=lora_config,
    args=training_args,
    tokenizer=tokenizer
)

trainer.train()
5️⃣ Save the Fine-Tuned Model
After training, save the fine-tuned model and tokenizer for future use:

python
Copy
Edit
trainer.model.save_pretrained("gemma-cuda-finetuned")
tokenizer.save_pretrained("gemma-cuda-finetuned")
6️⃣ Generate CUDA Code
Now that the model is fine-tuned, you can use it to generate CUDA code based on the input prefix and suffix.

python
Copy
Edit
def generate_cuda_code_fim(prefix, suffix):
    formatted_prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=500)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code
7️⃣ Model Deployment
Once you have the fine-tuned model, you can load it for inference and deploy it in any production system.

python
Copy
Edit
model = AutoModelForCausalLM.from_pretrained("gemma-cuda-finetuned")
tokenizer = AutoTokenizer.from_pretrained("gemma-cuda-finetuned")
⚙️ Example Usage
Here's a quick example to generate a CUDA snippet based on a given prefix and suffix:

python
Copy
Edit
test_prefix = """#include <cuda.h>
__global__ void add(int *a, int *b, int *c) {"""
test_suffix = """cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
add<<<1, 1>>>(d_a, d_b, d_c);"""
generated_code = generate_cuda_code_fim(test_prefix, test_suffix)

print("Generated CUDA Code:")
print(generated_code)
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Hugging Face: For providing the CodeGemma model and transformers library.

Google Colab: For providing a cloud environment for training.

LoRA: For enabling efficient and lightweight fine-tuning.

📬 Contact
Feel free to open an issue or contact me via email if you have any questions or suggestions!

💡 Happy Coding! 🎉
