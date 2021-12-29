from datasets import load_dataset
import numpy as np
import seaborn as sns
import torch
from torch import optim
from torch.nn import functional as F
from transformers import AdamW, MT5ForConditionalGeneration, MT5Tokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_len",default=20,type=int)
parser.add_argument("--model_path", default='mt5_translation_zh.pt', type=str)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--language", default='zh', type=str)

args = parser.parse_args()

print(args)

model_repo = 'google/mt5-small'
model_path = './model/'+args.model_path
max_seq_len = args.max_seq_len
device = torch.device(args.device)

tokenizer = MT5Tokenizer.from_pretrained(model_repo)
model = MT5ForConditionalGeneration.from_pretrained(model_repo)
model = model.to(device)

dataset = load_dataset('wmt17',args.language+'-en')

train_dataset = dataset['train']
test_dataset = dataset['test']

LANG_TOKEN_MAPPING = {
    'en': '<en>',
    'zh': '<zh>',
    'de' : '<de>'
}
special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

def encode_input_str(text, target_lang, tokenizer, seq_len,
                     lang_token_map=LANG_TOKEN_MAPPING):
  target_lang_token = lang_token_map[target_lang]

  # Tokenize and add special tokens
  input_ids = tokenizer.encode(
      text = target_lang_token + text,
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = True,
      max_length = seq_len)

  return input_ids[0]
  
def encode_target_str(text, tokenizer, seq_len,
                      lang_token_map=LANG_TOKEN_MAPPING):
  token_ids = tokenizer.encode(
      text = text,
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = True,
      max_length = seq_len)
  
  return token_ids[0]

def format_translation_data(translations, lang_token_map,
                            tokenizer, seq_len=128):
  # Choose languages for in i/o
  input_lang, target_lang = ['en', args.language]

  # Get the translations for the batch
  input_text = translations[input_lang]
  target_text = translations[target_lang]

  if input_text is None or target_text is None:
    return None

  input_token_ids = encode_input_str(
      input_text, target_lang, tokenizer, seq_len, lang_token_map)
  
  target_token_ids = encode_target_str(
      target_text, tokenizer, seq_len, lang_token_map)

  return input_token_ids, target_token_ids

def transform_batch(batch, lang_token_map, tokenizer):
  inputs = []
  targets = []
  for translation_set in batch['translation']:
    formatted_data = format_translation_data(
        translation_set, lang_token_map, tokenizer, max_seq_len)
    
    if formatted_data is None:
      continue
    
    input_ids, target_ids = formatted_data
    inputs.append(input_ids.unsqueeze(0))
    targets.append(target_ids.unsqueeze(0))

  batch_input_ids = torch.cat(inputs).to(device)
  batch_target_ids = torch.cat(targets).to(device)

  return batch_input_ids, batch_target_ids

def get_data_generator(dataset, lang_token_map, tokenizer, batch_size=32):
  dataset = dataset.shuffle()
  for i in range(0, len(dataset), batch_size):
    raw_batch = dataset[i:i+batch_size]
    yield transform_batch(raw_batch, lang_token_map, tokenizer)

# Constants
n_epochs = 8
batch_size = 8
print_freq = 50
checkpoint_freq = 1000
lr = 5e-4
n_batches = int(np.ceil(len(train_dataset) / batch_size))
total_steps = n_epochs * n_batches
n_warmup_steps = int(total_steps * 0.01)

# Optimizer
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, n_warmup_steps, total_steps)

losses = []

def eval_model(model, gdataset, max_iters=8):
  test_generator = get_data_generator(gdataset, LANG_TOKEN_MAPPING,
                                      tokenizer, batch_size)
  eval_losses = []
  for i, (input_batch, label_batch) in enumerate(test_generator):
    if i >= max_iters:
      break

    model_out = model.forward(
        input_ids = input_batch,
        labels = label_batch)
    eval_losses.append(model_out.loss.item())

  return np.mean(eval_losses)

for epoch_idx in range(n_epochs):
  # Randomize data order
  try:
    data_generator = get_data_generator(train_dataset, LANG_TOKEN_MAPPING,
                                      tokenizer, batch_size)
                
    for batch_idx, (input_batch, label_batch) \
        in tqdm(enumerate(data_generator), total=n_batches):
      optimizer.zero_grad()
    
    # Forward pass
      model_out = model.forward(
          input_ids = input_batch,
          labels = label_batch)

    # Calculate loss and update weights
      loss = model_out.loss
      losses.append(loss.item())
      loss.backward()
      optimizer.step()
      scheduler.step()

      # Print training update info
      if (batch_idx + 1) % print_freq == 0:
        avg_loss = np.mean(losses[-print_freq:])
        print('Epoch: {} | Step: {} | Avg. loss: {:.3f} | lr: {}'.format(
            epoch_idx+1, batch_idx+1, avg_loss, scheduler.get_last_lr()[0]))
      
      if (batch_idx + 1) % checkpoint_freq == 0:
        test_loss = eval_model(model, test_dataset)
        print('Saving model with test loss of {:.3f}'.format(test_loss))
        torch.save(model.state_dict(), model_path)
  except:
      continue

torch.save(model.state_dict(), model_path)

# Graph the loss
window_size = 50
smoothed_losses = []
for i in range(len(losses)-window_size):
  smoothed_losses.append(np.mean(losses[i:i+window_size]))

plt.plot(smoothed_losses[100:])
plt.savefig('loss_{}.png'.format(args.language))