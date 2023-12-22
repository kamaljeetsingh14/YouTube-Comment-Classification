import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy

from transformers import TFAutoModel, RobertaTokenizer

# Load RoBERTa tokenizer and model
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = TFAutoModel.from_pretrained('roberta-base')
def emb(examples):
    input = tokenizer(examples["text"],return_tensors="tf",max_length=64,padding="max_length",truncation=True)
    outputs=roberta(**input)
    return {"embeddings":outputs.last_hidden_state}
# use the tokenizer from DistilRoBERTa
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")


def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")


def to_bow(example):
    """Converts the sequence of 1-hot vectors into a single many-hot vector"""
    vector = numpy.zeros(shape=(tokenizer.vocab_size,))
    vector[example["input_ids"]] = 1
    return {"input_bow": vector}


def train(model_path="model", train_path="train.csv", dev_path="dev.csv"):

    # load the CSVs into Huggingface datasets to allow use of the tokenizer
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # the labels are the names of all columns except the first
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        return {"labels": [float(example[l]) for l in labels]}
    
    # convert text and labels to format expected by model
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    #hf_dataset = hf_dataset.map(to_bow)
    hf_dataset= hf_dataset.map(emb,batched=True)    #encoded_input = tokenizer(hf_dataset, return_tensors='tf')
    

    # convert Huggingface datasets to Tensorflow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="embeddings",
        label_cols="labels",
        batch_size=16,
        shuffle=True)
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="embeddings",
        label_cols="labels",
        batch_size=16)
    

    #print(train_dataset[1,])
    #print(numpy.shape(train_dataset))

    #inputs = tf.keras.Input(shape=(50265,), dtype=tf.int32, name='input_ids')

    # roberta_output = roberta(inputs)[0]  # Extracting the output embeddings

    #x = tf.keras.layers.Dense(32, activation='relu')(roberta_output[:, 0, :])
    #outputs = tf.keras.layers.Dense(len(labels), activation='sigmoid')(x)

    #model = tf.keras.Model(inputs=inputs, outputs=outputs)
        



    model = tf.keras.Sequential([
        #tf.keras.layers.Embedding(input_dim= tokenizer.vocab_size, output_dim=64),
        #
        #tf.keras.layers.Flatten(),
        tf.keras.Input(shape=(64,768)),
        tf.keras.layers.Bidirectional((tf.keras.layers.GRU(128,return_sequences=True))),
        #tf.keras.layers.Bidirectional((tf.keras.layers.GRU(128,return_sequences=True))),
        #tf.keras.layers.Bidirectional((tf.keras.layers.GRU(128,return_sequences=True))),
        tf.keras.layers.Bidirectional((tf.keras.layers.GRU(64,return_sequences=True))),
        tf.keras.layers.GlobalMaxPool1D(),
    #tf.keras.layers.Dense(units=256,input_dim=tokenizer.vocab_size,activation="relu"),
    #tf.keras.layers.Dense(units=128,activation='relu'),
        #tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None),
    tf.keras.layers.Dense(units=64,activation='relu'),
        #tf.keras.layers.Dense(units=128,activation='relu'),
        #tf.keras.layers.Dense(units=64,activation='relu'),
        #tf.keras.layers.Dense(units=128,activation='relu'),
        #tf.keras.layers.Dense(units=64,activation='relu'),
    tf.keras.layers.Dense(units=32,activation='relu'),
        #tf.keras.layers.Dense(units=32,activation='relu'),
    tf.keras.layers.Dense(units=16,activation='relu'),
    tf.keras.layers.Dense(units=len(labels),activation='sigmoid') 
    ])

    # specify compilation hyperparameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)])
    """early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_f1_score',
    baseline=0.79,
    min_delta=0.001,
    patience=3,
    restore_best_weights=True)"""
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor="val_f1_score",
    mode="max",
    save_best_only=True)

    model.fit(
    train_dataset,
    epochs=10,
    validation_data=dev_dataset,
    callbacks=[ model_checkpoint])



def predict(model_path="model", input_path="test-in.csv"):

    # load the saved model
    model = tf.keras.models.load_model(model_path)

    # load the data for prediction
    # use Pandas here to make assigning labels easier later
    df = pandas.read_csv(input_path)

    # create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset=hf_dataset.map(emb,batched=True)
    #hf_dataset = hf_dataset.map(to_bow)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="embeddings",
        batch_size=16)

    # generate predictions from model
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()
