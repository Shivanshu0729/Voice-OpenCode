import inspect
from livekit.plugins import cartesia
print('stream sig:', inspect.signature(cartesia.TTS.stream))
print('synthesize sig:', inspect.signature(cartesia.TTS.synthesize))
