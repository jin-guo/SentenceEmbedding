--[[

  Functions for loading data from disk.

--]]
function SentenceEmbedding.read_embedding(vocab_path, emb_path)
  local vocab = SentenceEmbedding.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

-- Read each sentences from the input file located in path
-- Map each token in sentence to the index of the vocab
function SentenceEmbedding.read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      -- print('Word:'..token)
      sent[i] = vocab:index(token)
    end
    -- print(sent)
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end

function SentenceEmbedding.read_corpus(dir, vocab)
  local corpus = {}
  corpus.vocab = vocab
  corpus.sentences = SentenceEmbedding.read_sentences(dir .. 'skipthough_sentence_content.txt', vocab)
  local sentence_id_file = torch.DiskFile(dir .. 'skipthough_sentence_id.txt')
  corpus.ids = {}
  for i = 1, #corpus.sentences do
    corpus.ids[sentence_id_file:readString("*l")] = i
  end
  sentence_id_file:close()
  return corpus
end


function SentenceEmbedding.read_skipthough_dataset(dir)
  local dataset = {}
  dataset.embedding_sentence = {}
  local embedding_sentence_id_file = io.open(dir .. 'embedding_sentence_id.txt')
  if embedding_sentence_id_file then
    for line in embedding_sentence_id_file:lines() do
      dataset.embedding_sentence[#dataset.embedding_sentence + 1] = line
    end
  end

  dataset.pre_sentence = {}
  local pre_sentence_id_file = io.open(dir .. 'pre_sentence_id.txt')
  if pre_sentence_id_file then
    for line in pre_sentence_id_file:lines() do
      dataset.pre_sentence[#dataset.pre_sentence + 1] = line
    end
  end

  dataset.post_sentence = {}
  local post_sentence_id_file = io.open(dir .. 'post_sentence_id.txt')
  if post_sentence_id_file then
    for line in post_sentence_id_file:lines() do
      dataset.post_sentence[#dataset.post_sentence + 1] = line
    end
  end

  embedding_sentence_id_file.close()
  pre_sentence_id_file.close()
  post_sentence_id_file.close()

  dataset.size = #dataset.embedding_sentence
  return dataset
end
