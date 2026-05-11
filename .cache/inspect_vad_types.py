import inspect
import livekit.agents.voice_activity_detection as vad
print('VADEventType:', getattr(vad, 'VADEventType', None))
if getattr(vad, 'VADEventType', None):
    print(list(vad.VADEventType))
    try:
        print('VADEvent attrs:', [x for x in dir(vad.VADEvent) if not x.startswith('_')])
    except Exception as e:
        print('VADEvent inspect error', e)
