import json
import pprint
import re
import nltk
import torch
from nltk.corpus import stopwords
from torch import optim, nn
from gpt import GPT

"""
Materials:
preprocessing text - https://medium.com/@pawan329/text-data-preprocessing-made-easy-steps-to-clean-text-data-using-python-81a138a0e0e3
basic model - https://medium.com/@sntaus/building-a-mini-gpt-like-language-model-from-scratch-27257bf5c145
dataset - https://rajpurkar.github.io/SQuAD-explorer/
"""

training_data = {}


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    return tokens


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


def perform_lemmatization(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def clean_text(text):
    tokens = preprocess_text(text)
    filtered_tokens = remove_stopwords(tokens)
    lemmatized_tokens = perform_lemmatization(filtered_tokens)
    clean_text = ' '.join(lemmatized_tokens)
    return clean_text


print("Preparing dataset")

with open("train.json") as f:
    templates = json.load(f)
    for i in templates["data"][0]["paragraphs"][0]["qas"]:
        x = clean_text(i['question'])
        y = clean_text(i['answers'][0]['text']) + ' <end>'
        training_data[x] = y

for item in training_data.items():
    print(f"{item[0]}: {item[1]}")

data_words = [k for k, _ in training_data.items()]
target_words = [v for _, v in training_data.items()]

vocabulary_words = list(
    set([element.lower() for nestedlist in [x.split(" ") for x in data_words] for element in nestedlist] + [
        element.lower() for nestedlist in [x.split(" ") for x in target_words] for element in nestedlist]))

vocabulary_words.remove("<end>")
vocabulary_words.append("<end>")
vocabulary_words.insert(0, "")

word_to_ix = {vocabulary_words[k].lower(): k - 1 for k in range(len(vocabulary_words))}
ix_to_word = {v: k for k, v in word_to_ix.items()}

vocab_size = len(word_to_ix)
embed_size = 512
num_layers = 4
heads = 3
learning_rate = 0.00001
epochs = 150
max_output_token_count = 25


def words_to_tensor(seq_batch, device=None):
    index_batch = []

    for seq in seq_batch:
        word_list = seq.lower().split(" ")
        indices = [word_to_ix[word] for word in word_list if word in word_to_ix]
        t = torch.tensor(indices)
        if device is not None:
            t = t.to(device)
        index_batch.append(t)

    return index_batch


def tensor_to_words(tensor):
    index_batch = tensor
    res = []

    for indices in index_batch:
        words = []
        for ix in indices:
            ix = ix.tolist()
            words.append(ix_to_word[ix].lower())
            if ix == word_to_ix["<end>"]:
                break

        res.append(' '.join(words))

    return res


def train(model, data, targets, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    batch_size = len(data)

    for b in range(batch_size):
        end_encountered = False
        cur_count = 0

        token_count = len(data[b])
        token_count_out = len(targets[b])

        while not end_encountered:
            target_vector = torch.zeros(model.vocab_size).to(device)

            if cur_count != token_count_out:
                expected_next_token_idx = targets[b].tolist()[cur_count]
                target_vector[expected_next_token_idx] = 1

            if cur_count > 0:
                model_input = data[b].reshape(token_count).to(device)
                part_of_output = torch.tensor(targets[b].tolist()[:cur_count]).to(device)
                model_input = torch.cat((model_input, part_of_output))
            else:
                model_input = data[b]

            out = model(model_input.reshape(1, token_count + cur_count))

            loss = criterion(out, target_vector.reshape(out.shape))
            total_loss += loss
            cur_count += 1

            if cur_count > token_count_out:
                end_encountered = True

    total_loss.backward()
    optimizer.step()
    return total_loss.item() / batch_size


def infer(model, input_vectors, max_output_token_count):
    model.eval()
    outputs = []

    for i in range(len(input_vectors)):
        input_vector = input_vectors[i].reshape(1, len(input_vectors[i]))
        predicted_sequence = []
        wc = 0

        with torch.no_grad():
            while True:
                output = model(input_vector)
                predicted_index = output[0, :].argmax().item()
                predicted_sequence.append(predicted_index)

                if predicted_index == word_to_ix['<end>'] or wc > max_output_token_count:
                    break

                input_vector = torch.cat([input_vector, torch.tensor([[predicted_index]])], dim=1)
                wc += 1

        outputs.append(torch.tensor(predicted_sequence))

    return outputs


device = torch.device("cpu")
model = GPT(vocab_size, embed_size, num_layers, heads).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("Tokenizing dataset")
data = words_to_tensor(data_words, device=device)
targets = words_to_tensor(target_words, device=device)

print("Training")
for epoch in range(epochs):
    avg_loss = train(model, data, targets, optimizer, criterion, device)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

torch.save(model, './save/model.pt')

input_vectors = words_to_tensor(data_words, device=device)
predicted_vector = infer(model, input_vectors, max_output_token_count)
predicted_words = tensor_to_words(predicted_vector)

print("\nTraining Data:")
pprint.pprint(training_data)

print("\nModel Inference:")
result_data = {data_words[k]: predicted_words[k] for k in range(len(predicted_words))}
pprint.pprint(result_data)

