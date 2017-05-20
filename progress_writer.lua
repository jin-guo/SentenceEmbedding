local progress_writer = torch.class('sentenceembedding.progress_writer')

function progress_writer:__init(config)
  self.progress_file_name = config.progress_file_name
  self.progress_file = io.open(self.progress_file_name, "w")
  print('Writing training progress to ' .. self.progress_file_name)
end

function progress_writer:write_skipthought_model_config(model)
  self.progress_file:write('--------------------------Model Configuration:--------------------------\n')
  self.progress_file:write(string.format('%-25s = %s\n',   'Model Type', model.name))
  self.progress_file:write(string.format('%-25s = %.2e\n', 'initial learning rate', model.learning_rate))
  self.progress_file:write(string.format('%-25s = %d\n',   'minibatch size', model.batch_size))
  self.progress_file:write(string.format('%-25s = %f\n',   'gradient clipping', model.grad_clip))
  self.progress_file:write(string.format('%-25s = %.2e\n',  'regularization strength', model.reg))
  self.progress_file:write('\n')
  self.progress_file:write(string.format('%-25s = %d\n',   'Word Embedding dim', model.emb_dim))
  self.progress_file:write(string.format('%-25s = %d\n',   'Word Embedding Update Flag', model.update_word_embedding))
  self.progress_file:write(string.format('%-25s = %s\n',   'Encoder structure', model.encoder_structure))
  self.progress_file:write(string.format('%-25s = %d\n',   'Encoder hidden dim', model.encoder_hidden_dim))
  self.progress_file:write(string.format('%-25s = %d\n',   'Encoder # of layers', model.encoder_num_layers))
  self.progress_file:write(string.format('%-25s = %d\n',   'Decoder hidden dim', model.decoder_hidden_dim))
  self.progress_file:write(string.format('%-25s = %d\n',   'Decoder # of layers', model.decoder_num_layers))
  self.progress_file:write('------------------------------------------------------------------------\n')

end

function progress_writer:write_fastsent_model_config(model)
  self.progress_file:write('--------------------------Model Configuration:--------------------------\n')
  self.progress_file:write(string.format('%-25s = %s\n',   'Model Type', model.name))
  self.progress_file:write(string.format('%-25s = %.2e\n', 'initial learning rate', model.learning_rate))
  self.progress_file:write(string.format('%-25s = %d\n',   'minibatch size', model.batch_size))
  self.progress_file:write(string.format('%-25s = %f\n',   'gradient clipping', model.grad_clip))
  self.progress_file:write(string.format('%-25s = %.2e\n',  'regularization strength', model.reg))
  self.progress_file:write('\n')
  self.progress_file:write(string.format('%-25s = %d\n',   'Word Embedding dim', model.emb_dim))
  self.progress_file:write('------------------------------------------------------------------------\n')

end

function progress_writer:write_contextencoder_model_config(model)
  self.progress_file:write('--------------------------Model Configuration:--------------------------\n')
  self.progress_file:write(string.format('%-25s = %s\n',   'Model Type', model.name))
  self.progress_file:write(string.format('%-25s = %.2e\n', 'initial learning rate', model.learning_rate))
  self.progress_file:write(string.format('%-25s = %d\n',   'minibatch size', model.batch_size))
  self.progress_file:write(string.format('%-25s = %f\n',   'gradient clipping', model.grad_clip))
  self.progress_file:write(string.format('%-25s = %.2e\n',  'regularization strength', model.reg))
  self.progress_file:write('\n')
  self.progress_file:write(string.format('%-25s = %d\n',   'Word Embedding dim', model.emb_dim))
  self.progress_file:write(string.format('%-25s = %d\n',   'Word Embedding Update Flag', model.update_word_embedding))
  self.progress_file:write(string.format('%-25s = %s\n',   'Encoder structure', model.encoder_structure))
  self.progress_file:write(string.format('%-25s = %d\n',   'Encoder hidden dim', model.encoder_hidden_dim))
  self.progress_file:write(string.format('%-25s = %d\n',   'Encoder # of layers', model.encoder_num_layers))
  self.progress_file:write('------------------------------------------------------------------------\n')
end
function progress_writer:write_string(string_to_write)
  self.progress_file:write(string_to_write)
end

function progress_writer:close_file()
  self.progress_file:close()
end
