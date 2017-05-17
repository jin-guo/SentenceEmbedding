--[[

  Training Script for Trace Software Artifacts.

--]]

require('..')

-- read command line arguments
local args = lapp [[
Training script for SkipThought on the Domain Document dataset.
  --encoder_layers (default 1)           	 Number of layers for Encoder
  --encoder_dim    (default 20)        	   Size of hidden dimension for Encoder
  --encoder_type   (default bigru)         Model Type for Encoder
  --decoder_layers (default 1)           	 Number of layers for Decoder
  --decoder_dim    (default 20)        	   Size of hidden dimension for Decoder
  --model_name     (default contextencoder)   Model Type to train
  --domain_name    (default ehr)           Domain name
  -u,--update_word_emb (default 0)         Update wordEmbedding flag
  -e,--epochs (default 10)                 Number of training epochs
  -r,--learning_rate (default 1.00e-02)    Learning Rate during Training NN Model
  -d,--learning_rate_decay (default 1)     Learning Rate Decay Flag
  -b,--batch_size (default 50)             Batch Size of training data point for each update of parameters
  -c,--grad_clip (default 5)               Gradient clip threshold
  -g,--reg  (default 1.00e-06)             Regulation lamda
  -o,--output_dir (default '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/') Output directory
  -w,--wordembedding_name (default 'healthIT_symbol_50d_w10_i20_word2vec') Name of the word embedding file
  -p,--progress_output (default 'progress_test.txt') Name of the progress output file
  -m,--model_output (default 'trained_model_test.model') Name of the trained model
]]


sentenceembedding.data_dir = args.output_dir .. 'data/'
sentenceembedding.models_dir = args.output_dir .. 'model/'
sentenceembedding.progress_dir = args.output_dir .. 'progress/'

if lfs.attributes(sentenceembedding.data_dir ) == nil then
  lfs.mkdir(sentenceembedding.data_dir )
end
if lfs.attributes(sentenceembedding.models_dir) == nil then
  lfs.mkdir(sentenceembedding.models_dir)
end
if lfs.attributes(sentenceembedding.progress_dir) == nil then
  lfs.mkdir(sentenceembedding.progress_dir)
end

local domain_name
if args.domain_name:lower() == 'ptc' then
  domain_name = 'PTC'
elseif args.domain_name:lower() == 'ehr' then
  domain_name = 'EHR'
else
  print('Unsupported Domain!')
  goto done
end

-- load embeddings
function load_word_embedding(vocab_path, emb_prefix)
  print('Loading word embeddings')
  local vocab = sentenceembedding.Vocab(vocab_path)
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
  return vocab, vecs
end

local vocab_path = sentenceembedding.data_dir..domain_name..'/Vocab.txt'
local emb_name = sentenceembedding.data_dir ..'wordEmbedding/' .. args.wordembedding_name
local vocab, vecs = load_word_embedding(vocab_path, emb_name)


local dataset_path = sentenceembedding.data_dir..domain_name..'/'
print('Loading Data for Domain: ' .. domain_name)

-- Read corpus and map each word in sentence to the index of the vocab
local corpus={}
corpus = sentenceembedding.read_corpus(dataset_path, vocab)
print('# of sentences in corpus:' .. #corpus.sentences)


-- Create dataset from the corpus
local dataset = {}
dataset = sentenceembedding.read_skipthought_dataset(dataset_path)
print('Data points in total:' .. #dataset.embedding_sentence)

-- Randomly split dataset to training and development set.
-- 1/k of the data goes to the development set, others to the training set
function split_dataset(dataset, k)
  if k>10 or k<1 then
    print('K should be between 1 to 10 for splitting dataset. The current input of K is: ', k)
    return
  end

  local train_set = {}
  local dev_set = {}
  train_set.embedding_sentence = {}
  train_set.pre_sentence = {}
  train_set.post_sentence = {}
  dev_set.embedding_sentence = {}
  dev_set.pre_sentence = {}
  dev_set.post_sentence = {}
  local bucket_size = math.floor(dataset.size/k)
  local indices = torch.randperm(dataset.size)
  for i = 1, dataset.size do
    local idx = indices[i]
    -- Copy data in the first bucket to development set.
    if i<=bucket_size then
      dev_set.embedding_sentence[i] = dataset.embedding_sentence[idx]
      dev_set.pre_sentence[i] = dataset.pre_sentence[idx]
      dev_set.post_sentence[i] = dataset.post_sentence[idx]
    else
      -- Copy data in the rest buckets to training set.
      train_set.embedding_sentence[i-bucket_size] = dataset.embedding_sentence[idx]
      train_set.pre_sentence[i-bucket_size] = dataset.pre_sentence[idx]
      train_set.post_sentence[i-bucket_size] = dataset.post_sentence[idx]
    end
  end
  train_set.size = #train_set.embedding_sentence
  dev_set.size = #dev_set.embedding_sentence
  return  train_set, dev_set
end

-- Initialze Progress writer
local progress_output_file_name = sentenceembedding.progress_dir..domain_name .. '_' .. args.progress_output
local progress_writer = sentenceembedding.progress_writer{
  progress_file_name = progress_output_file_name
}

-- Initialze SkipThought model
local model_class
if args.model_name:lower() == 'skipthought' then
  model_class = sentenceembedding.SkipThought
elseif args.model_name:lower() == 'autoencoder' then
  model_class = sentenceembedding.AutoEncoder
elseif args.model_name:lower() == 'contextencoder' then
  model_class = sentenceembedding.ContextEncoder
else
  print('Unsupported Modle Name!')
  goto done
end

local model
if args.model_name:lower() == 'contextencoder' then
  model = model_class{
    emb_vecs             = vecs,
    encoder_hidden_dim   = args.encoder_dim,
    encoder_num_layers   = args.encoder_layers,
    encoder_structure    = args.encoder_type,
    learning_rate        = args.learning_rate,
    batch_size           = args.batch_size,
    grad_clip            = args.grad_clip,
    reg                  = args.reg,
    update_word_embedding= args.update_word_emb,
    progress_writer      = progress_writer
  }
  progress_writer:write_contextencoder_model_config(model)
else
  model = model_class{
    emb_vecs             = vecs,
    encoder_hidden_dim   = args.encoder_dim,
    encoder_num_layers   = args.encoder_layers,
    encoder_structure    = args.encoder_type,
    decoder_hidden_dim   = args.decoder_dim,
    decoder_num_layers   = args.decoder_layers,
    learning_rate        = args.learning_rate,
    batch_size           = args.batch_size,
    grad_clip            = args.grad_clip,
    reg                  = args.reg,
    update_word_embedding= args.update_word_emb,
    progress_writer      = progress_writer
  }
  progress_writer:write_skipthought_model_config(model)
end


-- Number of epochs to train
local num_epochs = args.epochs

-- Print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()


local model_save_path = sentenceembedding.models_dir .. domain_name
  .. '_' .. args.model_output
local best_dev_loss = 1000000
-- Start Training
local train_set, dev_set = split_dataset(dataset, 10)
for i = 1, num_epochs do
  local start = sys.clock()
  print('----------------------------------------------------------------------\n')
  printf('-- epoch %d\n', i)
  printf('-- current learning rate %.6e\n', model.learning_rate)
  if model.learning_rate == 0 then
    break
  end
  printf('Start training model...\n')
  progress_writer:write_string(
    string.format('** %s %d **\n',   'Starting Epoch', i))
  progress_writer:write_string(
    string.format('** %s %.6e **\n',   'Current Learning Rate', model.learning_rate))

  local train_loss = model:train(train_set, corpus)
  printf('Start validating model...\n')
  local dev_loss = model:calcluate_loss(dev_set, corpus)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  printf('-- train loss: %.4f\n', train_loss)
  printf('-- dev loss: %.4f\n', dev_loss)

  progress_writer:write_string('***********************\n')
  progress_writer:write_string(
    string.format('%s %d in %.2f\n',   'Finished Epoch', i, sys.clock() - start))
  progress_writer:write_string(
    string.format('%s %.4f\n',   'Average Training Loss:', train_loss))
  progress_writer:write_string(
    string.format('%s %.4f\n',   'Average Development Loss:', dev_loss))
  progress_writer:write_string('***********************\n')
  if args.learning_rate_decay == 1 then
    model.learning_rate = model.learning_rate*0.1
  end

  if best_dev_loss > dev_loss then
    print('writing model for current epoch to ' .. model_save_path)
    model:save(model_save_path)
    best_dev_loss = dev_loss
  end
end
progress_writer:close_file()

 ::done::
