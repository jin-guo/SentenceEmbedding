package = "sentenceembedding"
version = "1.0-1"
source = {
   url = "..." -- We don't have one yet
}
description = {
   summary = "sentenceembedding",
   detailed = [[
      sentenceembedding
   ]],
   homepage = "http://...", -- We don't have one yet
   license = "MIT/X11" -- or whatever you like
}
dependencies = {
}
build = {
  -- type = "builtin",
  -- modules = {
  --   sentenceembedding = "init.lua",
  --   ["sentenceembedding.SkipThought"] = "SkipThought/SkipThought.lua",
  --   ["sentenceembedding.models.Encoder"] = "models/Encoder.lua",
  --   ["sentenceembedding.models.GRU"] = "models/GRU.lua",
  --   ["sentenceembedding.models.GRUDecoder"] = "models/GRUDecoder.lua",
  -- }
  type = "command",
build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"  -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" && $(MAKE)
]],
install_command = "cd build && $(MAKE) install"
}
