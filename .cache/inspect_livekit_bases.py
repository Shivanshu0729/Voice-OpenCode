import inspect
import livekit.agents.vad as vad_mod
import livekit.agents.tts as tts_mod

print('VAD module attrs:', [a for a in dir(vad_mod) if a in {'VAD', 'VADStream', 'VADEvent', 'VADEventType', 'VADCapabilities'}])
print('\nVAD source snippet:\n')
print('\n'.join(inspect.getsource(vad_mod).splitlines()[:260]))
print('\nTTS ChunkedStream source snippet:\n')
print('\n'.join(inspect.getsource(tts_mod.ChunkedStream).splitlines()[:220]))
