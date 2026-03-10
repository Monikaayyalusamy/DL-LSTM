# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset


## DESIGN STEPS
### STEP 1: 
Collect labeled text data for NER and preprocess it by tokenizing sentences, converting words to indices, and padding sequences.

### STEP 2: 
Convert words into dense vector representations using word embeddings.


### STEP 3: 
Design an LSTM network to capture sequential dependencies in the text.


### STEP 4: 
Add a dense layer with softmax activation to predict entity tags for each word.


### STEP 5: 
Train the model using labeled NER data and optimize using a suitable loss function.

### STEP 6: 
Evaluate the model using metrics like precision, recall, and F1-score, then use it to predict entities in new text.




## PROGRAM

### Name:Monika A

### Register Number:212224240094

```python


# Model definition
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100, dropout=0.1):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)           
        x = self.dropout(x)
        x, _ = self.lstm(x)         
        out = self.fc(x)     
        return out
        

vocab_size = len(word2idx) + 1   
tagset_size = len(tag2idx)

model = BiLSTMTagger(vocab_size=vocab_size, tagset_size=tagset_size).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_label_value)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)  # shape (batch, seq_len)
            optimizer.zero_grad()
            outputs = model(input_ids)          
            loss = loss_fn(outputs.view(-1, tagset_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, tagset_size), labels.view(-1))
                val_loss += loss.item()

        train_losses.append(total_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {total_loss:.4f} — Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

def evaluate_model(model, test_loader, X_test, y_test):
    model.eval()
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1)
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j] != tag2idx["O"]:
                        true_tags.append(idx2tag[labels[i][j].item()])
                        pred_tags.append(idx2tag[preds[i][j].item()])

# Run training and evaluation
train_losses, val_losses = train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3)
evaluate_model(model, test_loader, X_test, y_test)

# Plot loss
print('Name:Monika A')
print('Register Number:212224240094')
history_df = pd.DataFrame({"loss": train_losses, "val_loss": val_losses})
history_df.plot(title="Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Inference and prediction
i = 125
model.eval()
sample = X_test[i].unsqueeze(0).to(device)
output = model(sample)
preds = torch.argmax(output, dim=-1).squeeze().cpu().numpy()
true = y_test[i].numpy()

print('Name:Monika A')
print('Register Number:212224240094')
print("{:<15} {:<10} {}\n{}".format("Word", "True", "Pred", "-" * 40))
for w_id, true_tag, pred_tag in zip(X_test[i], y_test[i], preds):
    if w_id.item() != word2idx["ENDPAD"]:
        word = words[w_id.item() - 1]
        true_label = tags[true_tag.item()]
        pred_label = tags[pred_tag]
        print(f"{word:<15} {true_label:<10} {pred_label}")

```

### OUTPUT

## Loss Vs Epoch Plot

<img width="814" height="626" alt="Screenshot 2026-03-10 111340" src="https://github.com/user-attachments/assets/cf90bc44-5874-4031-b6c5-f941b336824b" />


### Sample Text Prediction

<img width="455" height="506" alt="image" src="https://github.com/user-attachments/assets/c217a4e9-1292-4cf2-8b16-ebac8dff0af9" />

## RESULT
The LSTM-based NER model successfully identifies named entities such as persons, locations, and organizations from text with good accuracy.
