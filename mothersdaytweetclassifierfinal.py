import csv
import random
import tensorflow as tf
sentences=[]
labels=[]
corpus=[]
stopwords=[ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
with open("/content/drive/My Drive/train.csv") as csvfile:
  csv_reader=csv.reader(csvfile)
  next(csv_reader)
  for rows in csv_reader:
    sentence=rows[1]
    for i in stopwords:
      word=" "+i+" "
      sentence=sentence.replace(word," ")
    corpus.append((sentence,int(rows[5])))
random.shuffle(corpus)
for i,j in corpus:
  sentences.append(i)
  labels.append(j)
print(sentences[0])
print((labels))

split=3000
train=sentences[:]
train_labels=[2 if(i==-1) else i for i in labels]
random.shuffle(corpus)
sentences=[]
labels=[]
for i,j in corpus:
  sentences.append(i)
  labels.append(j)
test=sentences[split:]
test_labels=[2 if(i==-1) else i for i in labels[split:]]
train_labels=tf.keras.utils.to_categorical(train_labels)
test_labels=tf.keras.utils.to_categorical(test_labels)
print(len(train),train_labels,len(test),len(test_labels))

max_len=32
total_size=12000
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer=Tokenizer(num_words=total_size)
tokenizer.fit_on_texts(train)
tokenizer.word_index={e:i for e,i in tokenizer.word_index.items() if i <= total_size}
print(total_size,tokenizer.word_index)
train_sequences=tokenizer.texts_to_sequences(train)
train_padded=pad_sequences(train_sequences,padding='post',truncating='post',maxlen=max_len)
print(train_padded.shape,max_len)
test_sequences=tokenizer.texts_to_sequences(test)
test_padded=pad_sequences(test_sequences,padding='post',truncating='post',maxlen=max_len)

import numpy as np
import zipfile
embedding_matrix={}
zip_ref = zipfile.ZipFile('/content/drive/My Drive/glove.twitter.27B.50d.zip', 'r') #download this glove 50dimensional tweet based pre-trained embedding vector from web
zip_ref.extractall('/tmp/')
zip_ref.close()
with open('/tmp/glove.twitter.27B.50d.txt') as embedding:
  for l in embedding:
    line=l.split()
    word=line[0]
    vect=np.asarray(line[1],dtype="float32")
    embedding_matrix[word]=vect
  embed_vect=np.zeros((total_size+1,50))
  for i,j in tokenizer.word_index.items():
    embed=embedding_matrix.get(i)
    if embed is not None:
      embed_vect[j]=embed

import tensorflow as tf
model=tf.keras.models.Sequential([
                                  tf.keras.layers.Embedding(total_size+1,50,input_length=max_len,weights=[embed_vect],trainable=True),
                                  tf.keras.layers.Dropout(0.1),
                                  tf.keras.layers.Conv1D(128,5,activation='relu',kernel_initializer='he_uniform'),
                                  tf.keras.layers.BatchNormalization(),
                                  tf.keras.layers.MaxPool1D(2),
                                  tf.keras.layers.Conv1D(64,3,activation='relu',kernel_initializer='he_uniform'),
                                  tf.keras.layers.BatchNormalization(),
                                  tf.keras.layers.MaxPool1D(2),
                                  tf.keras.layers.Conv1D(64,1,activation='relu',kernel_initializer='he_uniform'),
                                  tf.keras.layers.BatchNormalization(),
                                  tf.keras.layers.MaxPooling1D(2),
                                  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
                                  tf.keras.layers.LayerNormalization(),
                                  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
                                  tf.keras.layers.LayerNormalization(),
                                  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                                  tf.keras.layers.LayerNormalization(),
                                  tf.keras.layers.Dense(128,activation='relu',kernel_initializer='he_uniform'),
                                  tf.keras.layers.BatchNormalization(),
                                  tf.keras.layers.Dense(256,activation='relu',kernel_initializer='he_uniform'),
                                  tf.keras.layers.BatchNormalization(),
                                  tf.keras.layers.Dense(3,activation='softmax',kernel_initializer='he_uniform')
                                  ])
model.summary()

import numpy as np
tf.keras.backend.clear_session()
lr=tf.keras.callbacks.LearningRateScheduler(lambda x:1e-8 * 10**(x/20))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=tf.keras.metrics.Recall())
history=model.fit(train_padded,train_labels,epochs=100,batch_size=256,callbacks=[lr])

import matplotlib.pyplot as plt
plt.semilogx(history.history['lr'][80:],history.history['loss'][80:])

class Mycallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epochs,logs={}):
    if(logs.get('accuracy')>=0.95 and logs.get('recall')>=0.95):
      self.model.stop_training=True

callbacks=Mycallback()
tf.keras.backend.clear_session()
from tensorflow.keras.optimizers import Adam
recall=tf.keras.metrics.Recall()
precision=tf.keras.metrics.Precision()
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=6.5e-4),metrics=['accuracy',recall,precision ])
history=model.fit(train_padded,train_labels,validation_data=(test_padded,test_labels),epochs=1000,callbacks=[callbacks],batch_size=256)

model.load_weights('/content/drive/My Drive/savedweight2.hdf5') #You can train or just load my trained weights which i given in this repository to predict
model.evaluate(train_padded,np.array(train_labels),verbose=1)

acc=history.history['accuracy']
loss=history.history['loss']
val_acc=history.history['val_accuracy']
val_loss=history.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs,acc,label="acc")
plt.plot(epochs,val_acc,label="val_acc")
plt.legend()
plt.figure()
plt.plot(epochs,loss,label="loss")
plt.plot(epochs,val_loss,label="val_loss")
plt.legend()
plt.figure()

pred=[]
ids=[]
with open("/content/drive/My Drive/test (1).csv") as csvfile:
  csvread=csv.reader(csvfile,delimiter=',')
  next(csvread)
  
  for rows in csvread:
    pred.append(rows[1])
    ids.append(rows[0])
pred=tokenizer.texts_to_sequences(pred)
pred=pad_sequences(pred,maxlen=max_len,truncating="post",padding="post")
predictions=model.predict_classes(pred)
predictions=[-1 if(i==2) else i for i in predictions]
print(predictions,ids,sep='\n')
pred_dict={"id":ids,"sentiment_class":predictions}
import pandas as pd
ds=pd.DataFrame(pred_dict)
ds.to_csv('/content/drive/My Drive/prd.csv',index=False)

