require 'torch'
require 'nn'

require ('..')

function encoderTest()
  local emb_vecs = torch.Tensor({
    {1,2,1},
    {2,1,1},
    {2,2,2}
  })

  local config = {
    hidden_dim = 10,
    num_layers = 1,
    emb_vecs   = emb_vecs,
    structure  = 'bigru'
  }

  local sentence = torch.Tensor({
    {1,2,1},
    {2,1,1}
  })

  local gradOutputs = torch.Tensor({
    {1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1}
  })

  local encoder = sentenceembedding.Encoder(config)
  local forward_result = encoder:forward(sentence)
  print(forward_result)
  local backward_result = encoder:backward(sentence, gradOutputs)
  print(backward_result)

  return forward_result
end

function decoderTest(encoderOut)
  local emb_vecs = torch.Tensor({
    {1,2,1},
    {2,1,1},
    {2,2,2}
  })

  local config = {
    encoder_out_dim = 20,
    in_dim          = 3,
    hidden_dim      = 10,
    num_layers      = 2
  }

  local sentence = torch.Tensor({
    {1,2,1},
    {2,1,1},
    {1,2,1}
  })

  local gradOutputs = torch.Tensor({
    {1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1}
  })

  local decoder = sentenceembedding.GRUDecoder(config)
  local forward_result = decoder:forward(sentence, encoderOut)
  print(forward_result)
  local backward_result = decoder:backward(sentence, gradOutputs)
  print(backward_result)

  return forward_result
end

encoderOut = encoderTest()
encoderOut_cat = torch.cat(encoderOut[1], encoderOut[2], 1)
decoderTest(encoderOut_cat)
