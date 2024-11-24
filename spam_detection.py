import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from nltk.stem import PorterStemmer
from transformers import DataCollatorForSeq2Seq
from transformers import DataCollatorWithPadding
# Load dataset
df = pd.read_csv('spam.csv', encoding='latin1')

# Data preprocessing
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Label encoding target values
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicates
df = df.drop_duplicates(keep='first')

# Data visualization
plt.pie(df['target'].value_counts(), labels=['Ham', 'Spam'], autopct="%0.2f")
plt.title('Class Distribution')
plt.show()

# Add new features
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Visualize the distribution of number of characters
plt.figure(figsize=(12, 6))
sns.histplot(df[df['target'] == 0]['num_characters'], color='blue', label='Ham', kde=True)
sns.histplot(df[df['target'] == 1]['num_characters'], color='red', label='Spam', kde=True)
plt.legend()
plt.title('Number of Characters in Messages')
plt.show()

# Function for text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    text = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters
    text = [i for i in text if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation]  # Remove stopwords and punctuation
    text = [PorterStemmer().stem(i) for i in text]  # Apply stemming
    
    return " ".join(text)

# Apply preprocessing to the text column
df['transformed_text'] = df['text'].apply(transform_text)

# Create word clouds for Spam and Ham messages
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

# Wordcloud for spam
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.imshow(spam_wc)
plt.title("Spam WordCloud")
plt.axis('off')
plt.show()

# Wordcloud for ham
ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.imshow(ham_wc)
plt.title("Ham WordCloud")
plt.axis('off')
plt.show()

# Prepare dataset for BERT model
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['transformed_text'], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

train_dataset = Dataset.from_pandas(train_df[['transformed_text', 'target']])
test_dataset = Dataset.from_pandas(test_df[['transformed_text', 'target']])

# Rename 'target' column to 'labels' for compatibility with Trainer
train_dataset = train_dataset.rename_column("target", "labels")
test_dataset = test_dataset.rename_column("target", "labels")

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


# Use DataCollatorWithPadding for classification
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize Trainer with DataCollatorWithPadding
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Use test dataset as evaluation data
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Print evaluation metrics
print(results)

# Save the model and tokenizer
model.save_pretrained('./spam_classifier_model')
tokenizer.save_pretrained('./spam_classifier_model')

# Predict function using fine-tuned model
def predict(message):
    inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "Spam" if prediction == 1 else "Ham"

# Test the prediction function
message = "Congratulations, you've won a prize!"
print(predict(message))
