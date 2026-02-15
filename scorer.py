import torch 
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id="gpt2"
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id).to(device)
dataset= load_dataset("ibomohsin/gagle", streaming=True)
data_stream = dataset['train']
results = []
count =0
max_arts=100
#above line is used to split the dataset for easy loading and scoring
for entry in data_stream:
    if count > max_arts:
        break
    print(f"Available keys in this entry: {entry.keys()}")
    text = entry.get('llm_text') 
    
    if not text:
        continue
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] < 464:
        continue
# till here i have created the inital pipeline and the engine for scoring
# next we will take on the scoring logic 
    with torch.no_grad():
        outputs = model(input_ids)
        logits=outputs.logits
        # Shift logits and labels to align prediction with the next token
        shift_logits=logits[..., :-1, :].contiguous()
        shift_labels=input_ids[..., 1:].contiguous()
        # calculating the log probs
        log_probs=F.log_softmax(shift_logits, dim=-1)
        target_log_probs=torch.gather(log_probs, 2, shift_labels.unsqueeze(2)).squeeze(2)
        surprise_scores = target_log_probs[0].tolist()
        # basic parsing and refining of input dataset ie ignore first 64 tokens as a warmup and clip size to 400
        refined_scores = surprise_scores[64:464]
        # saving the scores in xlsx file
        results.append({
                "id": entry.get('_id'),
                "domain": entry.get('dataset_name'), # Maps to 'dataset_name' in your output
                "model_used": entry.get('model'),    # Maps to 'model' in your output
                "scores": refined_scores
            })
        count +=1
        print(f"processed {count}")
        with open("fractal_scores.json", "w") as f:
            json.dump(results, f)