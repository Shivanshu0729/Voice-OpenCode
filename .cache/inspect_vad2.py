import inspect
import livekit.agents.vad as av
print('VAD class:', av.VAD)
print('VAD.__init__ sig:', inspect.signature(av.VAD.__init__))
print('VAD methods:', [x for x in dir(av.VAD) if not x.startswith('_')])
