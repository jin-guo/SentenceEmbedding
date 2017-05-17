require('..')

-- read command line arguments
local args = lapp [[
Training script for SkipThought on the Domain Document dataset.
  --model_name     (default skipthought)   Model Type to train
  --domain_name    (default ehr)           Domain name
  -o,--output_dir (default '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/') Output directory
  -m,--model_output (default 'trained_model_test.model') Name of the trained model
]]

sentenceembedding.data_dir = args.output_dir .. 'data/'
sentenceembedding.models_dir = args.output_dir .. 'model/'
local model_name = args.model_name
local domain_name = args.domain_name
local model_file_name = args.model_output

local vocab = sentenceembedding.Vocab(sentenceembedding.data_dir..domain_name .. '/Vocab.txt')
local emb_path = sentenceembedding.data_dir .. domain_name .. '/'

-- Read corpus and map each word in sentence to the index of the vocab
local corpus={}
corpus = sentenceembedding.read_corpus(emb_path, vocab)
print('# of sentences in corpus:' .. #corpus.sentences)


-- Create dataset from the corpus
local dataset = {}
dataset = sentenceembedding.read_skipthought_dataset(emb_path)
print('Data points in total:' .. #dataset.embedding_sentence)

local gen = torch.Generator()
local sentence_idx = torch.random(gen, 1, #dataset.embedding_sentence)
print('Test Sentence ID: ', sentence_idx)




function test_skipthought_loss(dataset, corpus, sentence_idx, model)
  local embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx, post_sentence_with_vocab_idx =
    model:load_input_sentences(sentence_idx, dataset, corpus)
  if embedding_sentence_with_vocab_idx == nil or
    pre_sentence_with_vocab_idx == nil or
    post_sentence_with_vocab_idx == nil then
      print('Sentence Loading error for index:')
      print(idx)
      return
  end

  -- Initialze each sentence with its token mapped to embedding vectors
  local embedding_sentence, pre_sentence, post_sentence =
    model:input_module_forward(embedding_sentence_with_vocab_idx, pre_sentence_with_vocab_idx,
    post_sentence_with_vocab_idx)

  if embedding_sentence == nil or pre_sentence == nil or post_sentence == nil then
    print('Sentence too short.')
    return
  end

  -- Start the forward process
  local output_for_decoder = model:encoder_forward(embedding_sentence)
  print('Encoder Result Size')
  print(output_for_decoder:size())


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
    print(vocab:token(max_inx))
  end

  local sentence_loss = model.criterion:forward(decoder_output, target)
  print(sentence_loss)
  -- Important: to clear the grad_input from the last forward step.
  model.encoder:forget()
  model.decoder_pre:forget()
  model.decoder_post:forget()
end

function test_autoencoder_loss(dataset, corpus, sentence_idx, model)
  local embedding_sentence_with_vocab_idx =
    model:load_input_sentences(sentence_idx, dataset, corpus)
  if embedding_sentence_with_vocab_idx == nil then
    print('Sentence Loading error for index:')
    print(sentence_idx)
    return
  elseif embedding_sentence_with_vocab_idx:size(1)<2 then
    print('Sentence too short.')
    return
  end

  -- Initialze sentence with its token mapped to embedding vectors
  local embedding_sentence = model.input_module:forward(embedding_sentence_with_vocab_idx)
  -- Remove the last token (EOS) as the input for the decoder
  local embedding_sentence_for_decoder_input = embedding_sentence:sub(1,-2)

  -- Start the forward process
  local output_for_decoder = model:encoder_forward(embedding_sentence)

  -- Forward result to Decoder
  local decoder_result = model.decoder:forward(embedding_sentence_for_decoder_input, output_for_decoder)

  local decoder_output = model.prob_module:forward(decoder_result)

  -- Create the prediction target from the embedding sentences
  local target = embedding_sentence_with_vocab_idx:sub(2, -1)

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
    print(vocab:token(max_inx))
  end

  local sentence_loss = model.criterion:forward(decoder_output, target)
  print(sentence_loss)
  -- Important: to clear the grad_input from the last forward step.
  model.encoder:forget()
  model.decoder:forget()
end


local model
if model_name == 'skipthought' then
  model = sentenceembedding.SkipThought.load(sentenceembedding.models_dir .. model_file_name)
  test_skipthought_loss(dataset, corpus, sentence_idx, model)
elseif model_name == 'autoencoder' then
  model = sentenceembedding.AutoEncoder.load(sentenceembedding.models_dir .. model_file_name)
  test_autoencoder_loss(dataset, corpus, sentence_idx, model)
end
