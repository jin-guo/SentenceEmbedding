require('..')

sentenceembedding.data_dir = '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/data/'
sentenceembedding.models_dir = '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/model/'

local vocab = sentenceembedding.Vocab(sentenceembedding.data_dir..'healthIT_Vocab.txt')


-- Read corpus and map each word in sentence to the index of the vocab
local corpus={}
corpus = sentenceembedding.read_corpus(sentenceembedding.data_dir, vocab)
print('# of sentences in corpus:' .. #corpus.sentences)


-- Create dataset from the corpus
local dataset = {}
dataset = sentenceembedding.read_skipthough_dataset(sentenceembedding.data_dir)
print('Data points in total:' .. #dataset.embedding_sentence)

model = sentenceembedding.SkipThought.load(sentenceembedding.models_dir .. 'training_1.model')

local sentence_idx = 2
local embedding_sentence_with_vocab_idx = corpus.sentences[corpus.ids[dataset.embedding_sentence[sentence_idx]]]
print(embedding_sentence_with_vocab_idx)
if embedding_sentence_with_vocab_idx ~= nil then
  local forwardResult = model.input_module:forward(embedding_sentence_with_vocab_idx)
  local encoding_result = model:encoder_forward(forwardResult)
  print('Encoding result for sentence:', sentence_idx)
  print(encoding_result)
else
  print('Cannot load test sentence.')
end
