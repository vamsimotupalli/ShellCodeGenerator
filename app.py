from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from code_generator import read,tokenize,Encoder,Decoder,translate

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import os
import nltk
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	inputs,targets = read()
	input_tensor, inp_lang,maxlen_input,inputs = tokenize(inputs)
	target_tensor, targ_lang, maxlen_target,targets = tokenize(targets)
	
	BUFFER_SIZE = 3500
	BATCH_SIZE = 32
	
	input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.01)
	train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
	train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

	val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
	val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
	
	example_input_batch, example_target_batch = next(iter(train_dataset))
	example_input_batch.shape, example_target_batch.shape
	
	vocab_inp_size = len(inp_lang.word_index)+1
	vocab_tar_size = len(targ_lang.word_index)+1
	max_length_input = example_input_batch.shape[1]
	max_length_output = example_target_batch.shape[1]
	
	embedding_dim = 32
	units = 1024
	steps_per_epoch = 5

	encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
	# sample input
	sample_hidden = encoder.initialize_hidden_state()
	sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)

	decoder = Decoder(max_length_input,vocab_tar_size, embedding_dim, units, BATCH_SIZE, 'luong')
	sample_x = tf.random.uniform((BATCH_SIZE, max_length_output))
	decoder.attention_mechanism.setup_memory(sample_output)
	initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)


	sample_decoder_outputs = decoder(sample_x, initial_state,max_length_output)

	optimizer = tf.keras.optimizers.Adam()

	checkpoint_dir = 'D:\core_project\Code Generation'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                encoder=encoder,
                                decoder=decoder)
	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
	if request.method == 'POST':
		message = request.form['message']
		result = translate(str(message),inp_lang,max_length_input,units,encoder,targ_lang,decoder)
		print("123")
	return render_template('result.html',prediction = result)



if __name__ == '__main__':
	app.run(debug=True)