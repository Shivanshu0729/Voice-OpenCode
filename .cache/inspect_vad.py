import inspect
import livekit.agents.vad as av
print('VADStream init sig:', inspect.signature(av.VADStream.__init__))
print('VADStream dir:', [x for x in dir(av.VADStream) if not x.startswith('_')])
