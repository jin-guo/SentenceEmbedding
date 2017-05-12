require('..')

sentenceembedding.data_dir = '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/data/'
sentenceembedding.models_dir = '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/model/'

local vocab = sentenceembedding.Vocab(sentenceembedding.data_dir..'EHR/healthIT_Vocab.txt')


-- Read corpus and map each word in sentence to the index of the vocab
local corpus={}
corpus = sentenceembedding.read_corpus(sentenceembedding.data_dir .. 'EHR/', vocab)
print('# of sentences in corpus:' .. #corpus.sentences)


-- Create dataset from the corpus
local dataset = {}
dataset = sentenceembedding.read_skipthough_dataset(sentenceembedding.data_dir..'EHR/')
print('Data points in total:' .. #dataset.embedding_sentence)

model = sentenceembedding.SkipThought.load(sentenceembedding.models_dir .. 'alldata_learning_decay_every_epoch.model')

local sentence_idx = 2425
local embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx =
  model:load_input_sentences(sentence_idx, dataset, corpus)
if embedding_sentence_with_vocab_idx == nil or
  pre_sentence_with_vocab_idx == nil or
  post_sentence_with_vocab_idx == nil then
    print('Sentence Loading error for index:')
    print(idx)
    goto done
end

-- Initialze each sentence with its token mapped to embedding vectors
local embedding_sentence, pre_sentence, post_sentence =
  model:input_module_forward(embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx,
  post_sentence_with_vocab_idx)

if embedding_sentence == nil or pre_sentence == nil or post_sentence == nil then
  print('Sentence too short.')
  goto done
end

-- Start the forward process
local output_for_decoder = model:encoder_forward(embedding_sentence)

-- Forward result to Decoder
local decoder_result = model:decoder_forward(pre_sentence, post_sentence, output_for_decoder)

local decoder_output = model.prob_module:forward(decoder_result)
print('decoder_output:size()')
print(decoder_output:size())

-- Create the prediction target from the pre and post sentences
local pre_target, post_target
pre_target = pre_sentence_with_vocab_idx:sub(2, -1)
post_target = post_sentence_with_vocab_idx:sub(2, -1)
local target = torch.cat(pre_target, post_target, 1)
print('target:size()')
print(target:size())

local prediction = torch.exp(decoder_output)
for i=1, prediction:size(1) do
  local value_for_target = prediction[i][target[i]]
  print('Target:'..value_for_target)
  print(vocab:token(target[i]))
  local max = 0
  local max_inx = -1
  for j=1, prediction:size(2) do
    if prediction[i][j] > max then
      max = prediction[i][j]
      max_inx = j
    end
  end
  print('Prediction:' .. max)
  print('for index:' .. max_inx)
  print(vocab:token(max_inx))
end

local sentence_loss = model.criterion:forward(decoder_output, target)
print(sentence_loss)
-- Important: to clear the grad_input from the last forward step.
model.encoder:forget()
model.decoder_pre:forget()
model.decoder_post:forget()

 ::done::
