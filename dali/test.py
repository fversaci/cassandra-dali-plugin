import nvidia.dali.fn as fn
import nvidia.dali.plugin_manager as plugin_manager
plugin_manager.load_library('cpp/build/libcrs4cassandra.so')
help(fn.crs4_cassandra)
