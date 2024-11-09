import torch
import tiktoken
from layers import GPT2
from configs import GPTConfig
import torch.nn.functional as F

# Create model
model = GPT2(GPTConfig(vocab_size=50304))
model.eval()

# Load and tokenize input.txt
tokenizer = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
tokens = tokenizer.encode(text)

# Create small repeating batch for overfitting test
sequence_length = 32
batch_size = 4

buffer = tokens[:sequence_length * batch_size + 1]
buffer = torch.tensor(buffer, dtype=torch.long)
x = buffer[:-1]
y = buffer[1:]

# Reshape the tensors to have batch dimension
x = x.view(batch_size, sequence_length)
y = y.view(batch_size, sequence_length)

num_steps = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
for i in range(num_steps):
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {i}: | Loss: {loss.item()}")

model.eval()
num_return_sequences = 4
max_length = 32
tokens = tokenizer.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
xgen = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
sample_rng = torch.Generator()
sample_rng.manual_seed(42)
while xgen.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits, loss = model(xgen) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
        # gather the corresponding indices
    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    # append to the sequence
    xgen = torch.cat((xgen, xcol), dim=1)
# print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = tokenizer.decode(tokens)
    print(f"sample {i}: {decoded}")