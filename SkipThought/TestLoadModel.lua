require('..')

SentenceEmbedding.data_dir = '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/data/'
SentenceEmbedding.models_dir = '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/model/'

local vocab = SentenceEmbedding.Vocab(SentenceEmbedding.data_dir..'healthIT_Vocab.txt')


-- Read corpus and map each word in sentence to the index of the vocab
local corpus={}
corpus = SentenceEmbedding.read_corpus(SentenceEmbedding.data_dir, vocab)
print('# of sentences in corpus:' .. #corpus.sentences)


-- Create dataset from the corpus
local dataset = {}
dataset = SentenceEmbedding.read_skipthough_dataset(SentenceEmbedding.data_dir)
print('Data points in total:' .. #dataset.embedding_sentence)

model = SentenceEmbedding.SkipThought.load(SentenceEmbedding.models_dir .. '_1.model')
local train_loss = model:train(dataset, corpus)
