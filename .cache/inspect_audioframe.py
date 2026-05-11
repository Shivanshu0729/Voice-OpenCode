import inspect
from livekit import rtc
print('AudioFrame dir:', [x for x in dir(rtc.AudioFrame) if not x.startswith('_')])
try:
    import inspect as _inspect
    print('\nAudioFrame source snippet:\n')
    print('\n'.join(_inspect.getsource(rtc.AudioFrame).splitlines()[:200]))
except Exception as e:
    print('source err', e)
