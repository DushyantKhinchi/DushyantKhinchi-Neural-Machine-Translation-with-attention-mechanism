# DushyantKhinchi-Neural-Machine-Translation-with-attention-mechanism

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/csm_AdobeStock_261996669_fe16272c61.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)


## Goals

* To build a machine learning pipeline to translate one language to another (in this case from French to English). 
* Defining Seq2Seq model with many-to-many architecture.
* Defining the encoder layer as a combination of an embedding and an RNN layer(GRU in this case) to output hidden states for the computation of attention weights and the context vector.
* Defining the decoder layer as a combination of attention layer and fully connected layer.
* Building a helper function to store training weights as checkpoints. 

---

The dataset used in this project is a  corpus in a parallel corpus structure obtained from manythings.org

The corpus has 175623 lines and each line consists of an English word, phrase, or sentence and a french word, phrase, or sentence separated by a tab.


We started by building a class. This class maps words in a particular language to the index. This class has three dictionaries data structures.

word2int: to map each word to a unique index integer

int2word: to map index integer to the word 

word2count: to map a word to its total count in the corpus

functions that were used in the class

1) addWord() function just adds a word as a key to the word2int dictionary with its corresponding index value. The opposite is done for the int2word dictionary. It also keeps track of how many times we’ve come across a word when parsing our corpus to populate the class dictionaries and if we’ve already come across a particular word, we desist from adding it to the word2int and int2word dictionaries and instead keep track of how many times it appears in our corpus using the word2count dictionary.

2) addSentence() does is to simply iterate through each sentence and for each sentence, it splits the sentence into words and implements the addWord function on each word in each sentence.



 Corpus consists of French words which may have some characters like ‘Ç’. For simplicity sake, they were converted into their normal corresponding ASCII characters(Ç → C).

Also, spaces were created between words and punctuation attached to these words. (hello’s → hello s). This is to ensure that the presence of punctuation does not create two words for a particular word (For example, different integers would be assigned to “they’re” and "they are" although they have the same meaning).

To reduce the training time, we filtered out our dataset to remove sentences with more than ten words. Also, padding was applied to sentences with words less than the maximum length we proposed to cater for the batching process of our dataset. 

 

# Seq2Seq model
 

The many-to-many architecture was deployed which is suited for tasks such as Neural machine translation. This network is a combination of an encoder and a decoder. One to take in the input sentence and the other to translate into another language in the case of machine translation. This architecture can effectively handle the tasks where inputs and outputs have different lengths. 

The only information that the decoder receives from the encoder is the hidden state. The hidden state is a fixed size vector into which the encoder squeezes every bit of information about the input sentence and passes it on to the decoder to generate the output sentence. This might work fairly well for shorter sentences but this fixed size vector tends to become a bottleneck for longer sentences. This is where attention mechanism becomes a crucial part of our translation network.



## Encoder layer
Encoder unit is made up of two layers: the Embedding layer which converts each token into a dense representation and a Recurrent Network layer (here Gate Recurrent Unit Network has been used). 

The two very important parameters in the GRU implementation: return_sequences and return_state. return_sequences ensures that the GRU outputs the hidden state of each time step which we need to access information about each word in the input sequence during the computation of the attention weights and the context vector. Additionally, return_state returns the hidden state of the last time step. We need this tensor to be used as an initial hidden state for the decoder.

 

## Decoder layer
Decoder layer = Attention layer + fully connected layer

The attention layer is designed to return the context vector and the attention weights. In the attention layer The call function where forward propagation takes place takes in two parameters; encoder_out which represents all the hidden states at each timestep in the encoder and hidden which represents the hidden state of the decoder before the current timestep where we are generating the correct word and the score value is computed as follows. 

score=self.V(tf.nn.tanh(self.W1(encoder_out) + self.W2(hidden)))

And then this score value is passed through a softmax layer. The end product of the softmax function gives us the attention weights which we multiply with all the hidden states from the encoder. A hidden state at a particular timestep with a bigger weight value means more attention is being paid on the word at that timestep.

reduce_sum() will produce the context vector. After multiplying each hidden_state with its corresponding weight, we combine all resultant values through a summation which results in a context vector.

Tensor of a single word x is passed through an embedding layer that maps the single integer token into a dense 256-dimensional vector. That vector is later concatenated with the context vector generated by the attention layer. The resultant tensor becomes the input for the Gated Recurrent Network for a single timestep.

Finally, we pass the output of the GRU through a fully connected layer which outputs a vector of size (batch_size, number of English words).

We also return the hidden state to be fed into the next timestep and the attention weights for later visualizations.



Attention mechanism:

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/attn_model.png/:/rs=w:1280)

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/attn_mechanism.png/:/rs=w:1280)

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/image-11.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)


## Training pipeline
Starting with our loss function, Keras’s sparse categorical cross-entropy module was used since we have a large number of categories (number of english words). We create a mask that asserts that the padding tokens are not included in calculating the loss.

 

In a single training step, we pass the input_tensor which represents the input sentence through the forward propagation pipeline of the Encoder. This return the enc_output(hidden_state of all timesteps) and enc_hidden(last hidden_state).

In the decoding part, we use a technique called teacher forcing where instead of using the predicted word as input for the next time step, we use the actual word.

At the start of decoding, we feed the Start Of Sentence token as input and maximize the probability of the decoder predicting the first word in the output sequence as its output. We then take the actual first word and feed it into the second timestep and maximize the probability of the decoder predicting the second word in the output sequence as its output. This continues sequentially until we reach the End of Sentence token. We accumulate all the losses, derive the gradients, and train both networks end-to-end with the gradients.



A helper function called checkpoint was implemented is to save our model at certain points during our training. 

The final part of our training pipeline is a training loop. In each epoch, we iterate through our dataset and call the train_step function on each batch of the dataset. 10 epochs were run to train our model.

In order to perform the translation, we need to write a function much like what we did in the train_step function but instead of feeding in the actual word at a prior time step into the next time step, we feed in the word predicted by our network. This algorithm is known as Greedy search. 


![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/Screenshot_2.png/:/rs=w:1280)


## Translations
* "j'ai besoin de quelqu'un pour m'aider ?" was translated to " i need somebody to help me . EOS  "
* "ou vas tu ?" was translated to "where are you going ? EOS"
* "quand tu vas au marche ?" was translated to "when are you going to the market ? EOS"
---

Key Outcomes

• Built a machine learning pipeline to translate sentences from French to English.
• Parallel corpus structure of 175623 examples was used to train the model.
• Built a Seq2Seq model with many-to-many architecture.
• An encoder layer was implemented as a combination of an embedding and an RNN layer (GRU in this case) to output hidden states for the computation of attention weights and the context vector.
• A decoder layer was implemented as a combination of the attention layer and the fully connected layer.
• Implemented a helper function to store training weights as checkpoints.
• Bilingual evaluation understudy (Bleu) score was used as an evaluation metric.

• Average Bleu score for a test document with 5000 sentences was computed to be 0.263
