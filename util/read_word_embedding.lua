require('init')

opt = {
	binfilename = '/Users/Jinguo/Dropbox/TraceNN_experiment/skipthoughts/data/wordEmbedding/healthIT_w10_50d_20iter.txt',
	outVecs = sentenceembedding.data_dir .. '/wordembedding/healthIT_symbol_50d_w10_i20_word2vec.vecs',
  outVocab = sentenceembedding.data_dir .. '/wordembedding/healthIT_symbol_50d_w10_i20_word2vec.vocab'
}

--Reading the size
local count = 0 -- 0 for word2vec output file, 1 for glove
local dim = -1
for line in io.lines(opt.binfilename) do
    if count == 1 then
        for i in string.gmatch(line, "%S+") do
            dim = dim + 1
        end
    end
    count = count + 1
end
count = count-1

print("Reading embedding file with ".. count .. ' words of ' .. dim .. ' dimensions.' )


local emb_vecs = {}
local emb_vocab = {}
--Reading Contents

local i = 0 -- 0 for word2vec output file,  1 for glove
local word_count = 0
for line in io.lines(opt.binfilename) do
	if(i > 0) then
	  xlua.progress(i,count)
	  local vecrep = {}
	  for i in string.gmatch(line, "%S+") do
	    table.insert(vecrep, i)
	  end
	  str = vecrep[1]
	  table.remove(vecrep,1)
		vecrep = torch.DoubleTensor(vecrep)
		local norm = torch.norm(vecrep,2)
		if norm ~= 0 then vecrep:div(norm) end
		word_count = word_count+1
		emb_vecs[word_count] = vecrep
		emb_vocab[word_count] = str
	end
  i = i + 1
end
collectgarbage()

print('Start Writing Vectors File.')
torch.save(opt.outVecs,emb_vecs)
emb_vecs = nil
collectgarbage()

print('Start Writing Vobab File.')
local vocab_file = torch.DiskFile(opt.outVocab, 'w')
for i = 1, word_count do
  vocab_file:writeString(emb_vocab[i])
  if(i<count) then vocab_file:writeString('\n') end
end
vocab_file:close()
