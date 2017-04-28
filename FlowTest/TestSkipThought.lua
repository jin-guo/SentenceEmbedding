--[[

  Training Script for Trace Software Artifacts.

--]]

require('..')

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the TRACE dataset.
  --encoder_layers (default 1)           	 Number of layers for Encoder
  --encoder_dim    (default 20)        	   Size of hidden dimension for Encoder
  --encoder_type   (default bigru)           Model Type for Encoder
  --decoder_layers (default 1)           	 Number of layers for Decoder
  --decoder_dim    (default 20)        	   Size of hidden dimension for Decoder
  -e,--epochs (default 100)                 Number of training epochs
  -r,--learning_rate (default 1.00e-02)    Learning Rate during Training NN Model
  -b,--batch_size (default 1)              Batch Size of training data point for each update of parameters
  -c,--grad_clip (default 1)             Gradient clip threshold
  -g,--reg  (default 0)             Regulation lamda
  -t,--test_model (default false)          test model on the testing data
  -o,--output_dir (default '/home/lslc/Dropbox/TraceNN_experiment/skipthoughts/') Output directory
  -w,--wordembedding_name (default 'healthIT_symbol_50d_w10_i20_word2vec') Name of the word embedding file
  -p,--progress_output (default 'progress') Name of the progress output file
]]

sentenceembedding.data_dir = '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/data/'
sentenceembedding.models_dir = '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/model/'
-- load embeddings
print('Loading word embeddings')
local vocab = sentenceembedding.Vocab(sentenceembedding.data_dir..'healthIT_Vocab.txt')
local emb_file_name = args.wordembedding_name --'wiki_ptc_symbol_300d_w10_i10_word2vec'
local emb_dir = sentenceembedding.data_dir ..'wordembedding/'
local emb_prefix = emb_dir .. emb_file_name
local emb_vocab, emb_vecs = sentenceembedding.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.vecs')
local emb_dim
for i, vec in ipairs(emb_vecs) do
  emb_dim = vec:size(1)
  break
end
print('Embedding dim:', emb_dim)
print('vocabulary size:', vocab.size)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    -- print(w)
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('Unfound token count: unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()


-- Read corpus and map each word in sentence to the index of the vocab
local corpus={}
corpus = sentenceembedding.read_corpus(sentenceembedding.data_dir, vocab)
print('# of sentences in corpus:' .. #corpus.sentences)


-- Create dataset from the corpus
local dataset = {}
dataset = sentenceembedding.read_skipthough_dataset(sentenceembedding.data_dir)
print('Data points in total:' .. #dataset.embedding_sentence)


local model_class = sentenceembedding.SkipThought
local model = model_class{
  emb_vecs             = vecs,
  encoder_hidden_dim   = args.encoder_dim,
  encoder_num_layers   = args.encoder_layers,
  encoder_structure    = args.encoder_type,
  decoder_hidden_dim   = args.decoder_dim,
  decoder_num_layers   = args.decoder_layers,
  learning_rate        = args.learning_rate,
  batch_size           = args.batch_size,
  grad_clip            = args.grad_clip,
  reg                  = args.reg
}


-- Number of epochs to train
local num_epochs = args.epochs

-- Print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- Start Training
for i = 1, num_epochs do

  local start = sys.clock()
  printf('-- epoch %d\n', i)
  printf('-- current learning rate %.10f\n', model.learning_rate)
  local train_loss = model:train(dataset, corpus)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  printf('-- train loss: %.4f\n', train_loss)
end

local model_save_path = sentenceembedding.models_dir .. 'training_1.model'
print('writing model to ' .. model_save_path)
model:save(model_save_path)
