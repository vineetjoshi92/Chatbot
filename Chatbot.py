# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 12:24:04 2018

@author: Vineet Joshi
"""
# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time

# Importing the dataset
lines = open("movie_lines.txt", encoding = "utf-8", errors = 'ignore').read().split('\n')
conversations = open("movie_conversations.txt", encoding = "utf-8", errors = 'ignore').read().split('\n')

# Creating a dictionry to map each line to its id
id2line = {}

for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
# Creating a list of the conversations
conversations_id = []

for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_id.append(_conversation.split(","))
    
# Separating the questions and answers 
# For DIGITALMAN, I need to to just this.
questions = []
answers = []

for conversation in conversations_id:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Text cleaning

def clean_text(text):
    text = text.lower() # Lower case
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>?,{}+=~|.]", "", text)
    return text

# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
# Creating a dictionary to map words to their count
word2count = {}

for question in clean_questions:
    for word in question.split():
        if word in word2count:
            word2count[word] += 1
        else:
            word2count[word] = 1

for answer in clean_answers:
    for word in answer.split():
        if word in word2count:
            word2count[word] += 1
        else:
            word2count[word] = 1

# Creating two dictionaries to map words in questions and answers to a
# unique integer

threshold = 20
questionswords2int = {}
word_num = 0

for word, count in word2count.items():
    if count > threshold:
        questionswords2int[word] = word_num
        word_num += 1


answerswords2int = {}
word_num = 0

for word, count in word2count.items():
    if count > threshold:
        answerswords2int[word] = word_num
        word_num += 1

# Adding special tokens
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
    
# Creating inverse dictionary to map unique integers to words
answersints2word = {v:k for k, v in answerswords2int.items()} 

# Adding EOS token to end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
# Transforming all questions and answers to their unique integers
# Replacing all filtered out words with '<OUT>'

questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
            
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

# Sorting questions and answers by the length of the questions
# This speeds up the training because it will reduce the amount of padding during
# training.

sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1,26):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            
# BUILDING THE SEQ2SEQ MODEL

# In tensorflow, all variables are used in tensors.
# All the variables used in tensors must be defined as tensorflow placeholders.

# Creating placeholders for inputs and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    # Placeholders for learning rate and keep_prob which is used to control the 
    # dropout rate of neurons
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    return inputs, targets, lr, keep_prob


# Preprocessing the targets

# We have to do this because the decoder will only accept certain format for 
# the target. This format is two-fold.
# 1. The targets must be in batches. The RNN of the decoder will not accept 
# single targets.
# 2. Each of the answer in the batches MUST start with the <SOS> token.
# We will do this in the function below.

def preprocess_targets(targets, word2int, batch_size):
    '''left_side be a matrix of size batch_size X 1 and contain the unique integer
    associated with the SOS tag
    right_side will be a matrix of batch_size X and would get the unique integers 
    in the answer exlcuding the last token (which is the <EOS> token)'''
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    
    # tf.strided slice extracts a subset of a tensor
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    
    preprocessed_targets = tf.concat([left_side, right_side], axis = 1)
    return preprocessed_targets

# Creating the encoder RNN layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    '''rnn_inputs correspond to the model_inputs
    rnn_size is the number of input tensors in the encoder RNN layer
    sequence_length is the length of each question in the batch
    num_layers and keep_prob are hyperparameters of the model'''
    # Creates a LSTM using the BasicLSTMCell class
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size) 
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    # Creating the encoder cell
    # Encoder cell consists of several LSTM layers and we apply dropout to each layer
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    
    # Creates a dynamic version of a bidirectional rnn
    # Takes input and builds independent forward and backward RNNs
    # We have to make sure the input size of the forward cell and backward cell must match
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state
    
# Creating the Decoder RNN layer

# Step 1:
# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length,
                        decoding_scope, output_function, keep_prob, batch_size):
    ''' Deocder gets the encoder_state as input
    decoder_cell - Cell in the RNN of the decoder 
    Check tensorflow documentation for the other parameters
    output_function - the function we will use to return the decoder outputs'''
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                        attention_option = 'bahdanau',
                                                                        num_units = decoder_cell.output_size)
    
    # attention_keys-keys to be compared with the target state
    # attention_values = values that we will use to construct the context vector
    # context is returned by the encoder and that should be used by the decoder as the
    # first element of the decoding.
    # attention_score_function - used to compute the similarity between the keys and
    # the target states.
    # attention_construct_function is a function used to build the attention state
    
    # Decode the training set
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys, 
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function, 
                                                                              name = "attn_dec_train")
    # Get the decoder output
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# Step 2
# Decoding the validation set

def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length,
                    num_words, decoding_scope, output_function, keep_prob, batch_size):
    ''' Deocder gets the encoder_state as input
    decoder_cell - Cell in the RNN of the decoder 
    Check tensorflow documentation for the other parameters
    output_function - the function we will use to return the decoder outputs'''
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                        attention_option = 'bahdanau',
                                                                        num_units = decoder_cell.output_size)
    
    # Decode the training set
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys, 
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix, 
                                                                              sos_id, 
                                                                              eos_id, 
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    # Get the decoder output
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              test_decoder_function,
                                                                                                              scope = decoding_scope)
    
    return test_predictions

# Creating the decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state,
                num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob,
                batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        
        # Initializing the weights
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None, 
                                                                      scope = decoding_scope,
                                                                      weights_initializer=weights,
                                                                      biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input, 
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        
    return training_predictions, test_predictions 

# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, 
                  answers_num_words, questions_num_words, encoder_embedding_size, 
                  decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words+1,
                                                              encoder_embedding_size,
                                                              initializer=tf.random_uniform_initializer(0, 1))
    
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob,
                                sequence_length)
    
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size],0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, 
                                                         encoder_state, questions_num_words, sequence_length,
                                                         rnn_size, num_layers, questionswords2int, 
                                                         keep_prob, batch_size)
    
    return training_predictions, test_predictions
        

# SETTING THE HYPERPARAMETERS
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.03
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()
    
# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

# Getting shape of the inputs tensor
input_shape = tf.shape(inputs) 

# Creating the training and test predictions     
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets, keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size, num_layers,
                                                       questionswords2int)

# Setting up the loss error, the optimizer, and gradient clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)   

# Padding the sequences with the '<PAD>' token
# All the sentences in a batch, whether they are questions or answers must have
# the same length
# For each batch, and for each sentence in each batch, we will add PAD tokens so
# that each sentence of the batch has the same length.
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions)//batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
        
# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions)* 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# Training

batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1,epochs+1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], 
                                                   {inputs:padded_questions_in_batch,
                                                    targets:padded_answers_in_batch,
                                                    lr: learning_rate, 
                                                    sequence_length: padded_answers_in_batch.shape[1],
                                                    keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,epochs, batch_index, 
                  len(training_questions) // batch_size,
                  total_training_loss_error / batch_index_check_training_loss, 
                  int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, 
                                                   {inputs:padded_questions_in_batch,
                                                    targets:padded_answers_in_batch,
                                                    lr: learning_rate, 
                                                    sequence_length: padded_answers_in_batch.shape[1],
                                                    keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print('Sorry, I do not speak better, I need to practice more.')
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print('My apologies, I cannot speak better anymore. This is the best I can do.')
        break
print("Game Over!")
                
                
                
            

        
        
        