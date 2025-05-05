# 🚀 CodeGemma CUDA Fine-Tuning Project

Welcome to the **CodeGemma CUDA Fine-Tuning** repository!  
This project shows how to fine-tune the **CodeGemma** model for generating CUDA code snippets using your own dataset.  
With **LoRA (Low-Rank Adaptation)**, you can train efficiently without massive hardware.

---

## 💡 Key Features

✅ Fine-tune CodeGemma for CUDA code generation  
✅ Efficient, lightweight LoRA-based training  
✅ Generate CUDA code from custom prefixes and suffixes  

---

## 📦 Requirements

Install the necessary libraries:

```bash
pip install transformers datasets peft trl torch
```

| Library         | Purpose                                    |
|-----------------|-------------------------------------------|
| transformers    | For model handling                        |
| datasets        | For dataset loading and processing        |
| peft           | For Low-Rank Adaptation (LoRA)            |
| trl            | For streamlined fine-tuning with transformers |
| torch          | For PyTorch operations and GPU acceleration |
| json (builtin) | To handle JSON data                       |

---

## 📁 Project Structure

```
.
├── finetuning.py         # Script for fine-tuning and generation
├── cuda_dataset.json     # Custom CUDA dataset (replace with yours)
├── requirements.txt      # List of dependencies
├── LICENSE               # MIT License
└── README.md             # This file
```

---

## 🛠️ How to Run

### 1️⃣ Upload Your Custom Dataset

Prepare a JSON dataset like:

```json
[
  {
    "fim_text": "<fim_prefix>#include <stdio.h>\n#include <cuda.h>\n\n__global__ void add(int *a, int *b, int *c) { ... }<fim_suffix>"
  },
  {
    "fim_text": "<fim_prefix>int main(void) { ... }<fim_suffix>"
  }
]
```

Each item should include `"fim_text"` with your prefix, suffix, and middle markers.

---

### 2️⃣ Load the Pre-trained CodeGemma Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/codegemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

---

### 3️⃣ Preprocess the Dataset

```python
import json

def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [{"text": item["fim_text"]} for item in data]

formatted_data = load_json_dataset("cuda_dataset.json")
```

---

### 4️⃣ Fine-Tune with LoRA

```python
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
    train_dataset=formatted_data,  # make sure it's wrapped as a Dataset object
    eval_dataset=None,
    dataset_text_field="text",
    peft_config=lora_config,
    args=training_args,  # define your TrainingArguments separately
    tokenizer=tokenizer
)

trainer.train()
```

---

### 5️⃣ Save the Fine-Tuned Model

```python
trainer.model.save_pretrained("gemma-cuda-finetuned")
tokenizer.save_pretrained("gemma-cuda-finetuned")
```

---

### 6️⃣ Generate CUDA Code

```python
def generate_cuda_code_fim(prefix, suffix, model, tokenizer, device="cuda"):
    formatted_prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=500)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

# Example:
test_prefix = """#include <cuda.h>
__global__ void add(int *a, int *b, int *c) {"""
test_suffix = """cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
add<<<1, 1>>>(d_a, d_b, d_c);"""

generated_code = generate_cuda_code_fim(test_prefix, test_suffix, model, tokenizer)
print("Generated CUDA Code:")
print(generated_code)
```

---

### 7️⃣ Deploy for Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gemma-cuda-finetuned")
tokenizer = AutoTokenizer.from_pretrained("gemma-cuda-finetuned")
```

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **Hugging Face**: For providing the CodeGemma model and `transformers` library  
- **Google Colab**: For free cloud GPUs for training  
- **LoRA**: For enabling efficient fine-tuning

---

## 📬 Contact

Feel free to open an issue or reach out via email if you have questions or suggestions!

💡 **Happy CUDA coding! 🚀**
