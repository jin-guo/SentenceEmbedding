require('..')


-- load embeddings
print('Loading word embeddings')
local vocab = sentenceembedding.Vocab('/Users/Jinguo/Dropbox/TraceNN_experiment/tracenn/data/artifact/symbol/' .. 'vocab_ptc_artifact_clean.txt')

local emb_file_name = 'wiki_ptc_symbol_300d_w10_i10_word2vec'
local emb_dir = '/Users/Jinguo/Dropbox/TraceNN_experiment/tracenn/data/wordembedding/'
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


-- Read corpus and map each sentence to the index of the vocab
local corpus={}
corpus.sentences = sentenceembedding.read_sentences('/Users/Jinguo/Dropbox/LS_LC/Project/TSE/Data/sentence.txt', vocab)

for i = 1, #corpus.sentences do
  local sentence = corpus.sentences[i]
  print('Token count:'.. sentence:size(1))
  -- print(vecs:index(1, sentence:long()))  -- local src_artf = artifact.src_artfs[i]
  -- artifact.src_artfs[i] = vecs:index(1, src_artf:long())
end

local dataset = {}
dataset.embedding_sentence = {}
dataset.pre_sentence = {}
dataset.post_sentence = {}
for i = 2, #corpus.sentences-1 do
  dataset.embedding_sentence[#dataset.embedding_sentence + 1] =i
  dataset.pre_sentence[#dataset.embedding_sentence] =i
  dataset.post_sentence[#dataset.embedding_sentence] =i
end

print('Data points in total:' .. #dataset.embedding_sentence)

-- initialize SkipThought model
local encoder_config = {
  hidden_dim = 10,
  num_layers = 1,
  emb_vecs   = vecs,
  structure  = 'gru'
}
local decoder_config = {
  encoder_out_dim = 10,
  in_dim          = vecs:size(2),
  hidden_dim      = 10,
  num_layers      = 10
}
--


input_model = nn.Sequential()
local lookup = nn.LookupTable(vecs:size(1), vecs:size(2))
lookup.weight:copy(vecs)
input_model:add(lookup)

prob_module = nn.Sequential()
  :add(nn.Linear(decoder_config.hidden_dim, vecs:size(1)))
  :add(nn.LogSoftMax())

encoder = sentenceembedding.Encoder(encoder_config)
decoder_pre  = sentenceembedding.GRUDecoder(decoder_config)
decoder_post = sentenceembedding.GRUDecoder(decoder_config)

-- For getting all the parameters for the SkipThought model
st_modules = {}
local modules = nn.Parallel()
  :add(encoder)
  :add(decoder_pre)
  :add(prob_module)
st_modules.params, st_modules.grad_params = modules:getParameters()
self.encoder_params = encoder:parameters()

idx = 1
if corpus.sentences[dataset.embedding_sentence[idx]]~= nil then
  embedding_sentence = input_model:forward(corpus.sentences[dataset.embedding_sentence[idx]])
else
  print('Cannot find embedding sentence for current training data point:', dataset.embedding_sentence[idx])
end

if corpus.sentences[dataset.pre_sentence[idx]]~= nil then
  pre_sentence = input_model:forward(corpus.sentences[dataset.pre_sentence[idx]])
else
  print('Cannot find the sentence before the embedding sentence for current training data point:', dataset.pre_sentence[idx])
end

if corpus.sentences[dataset.post_sentence[idx]]~= nil then
  post_sentence = input_model:forward(corpus.sentences[dataset.post_sentence[idx]])
else
  print('Cannot find the sentence after the embedding sentence for current training data point:', dataset.post_sentence[idx])
end

-- Start forward process
encode_result = encoder:forward(embedding_sentence)
print(encode_result)

pre_decoder_result = decoder_pre:forward(pre_sentence,encode_result)
if decoder_config.num_layers == 1 then
  pre_final = prob_module:forward(pre_decoder_result)
  post_final = prob_module:forward(post_decoder_result)
else
  pre_final = prob_module:forward(pre_decoder_result:select(3, pre_decoder_result:size(3)))
  post_final = prob_module:forward(post_decoder_result:select(3, pre_decoder_result:size(3)))
end
-- final = prob_module:forward(pre_decoder_result)

-- Remove the last output since the EOS should be predicted before this output
pre_final = pre_final:sub(1, pre_final:size(1)-1)
print('Pre Sentence Final Result')
print(pre_final:size())

-- The prediction target starts from the second of the token in the sentence sequence
target_index = corpus.sentences[dataset.pre_sentence[idx]]:sub(2, corpus.sentences[dataset.pre_sentence[idx]]:size(1))
-- print('Target Index')
-- print(target_index:size())
--
-- -- Accumulate the loss for every prediction for the sentence
-- total_loss = 0
-- for i = 1, target_index:size(1) do
--   loss = pre_final[i][target_index[i]]
--   print(loss)
--   total_loss = total_loss + loss
-- end

-- Using the Negative Log Likelihood criterion.
criterion = nn.ClassNLLCriterion()
err = criterion:forward(pre_final, target_index)

print('LOSS: '.. err)

local crit_grad = criterion:backward(pre_final, target_index)

if decoder_config.num_layers == 1 then
  prob_grad = prob_module:backward(pre_decoder_result, crit_grad)
else
  prob_grad = prob_module:backward(pre_decoder_result:select(3, pre_decoder_result:size(3)), crit_grad)
end

local pre_decoder_input_grad, encoder_output_grads = decoder_pre:backward(pre_sentence, prob_grad)
print(encoder_output_grads)

local encode_grad = encoder:backward(embedding_sentence, encoder_output_grads)
