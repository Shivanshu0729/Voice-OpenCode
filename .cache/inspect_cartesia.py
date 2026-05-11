from livekit.plugins import cartesia
import inspect
print('module attrs:', [a for a in dir(cartesia) if 'Stream' in a or 'Synthesize' in a])
print('\nTTS attrs:', [a for a in dir(cartesia.TTS) if not a.startswith('_')])
try:
    import inspect
    print('\nTTS source snippet:\n')
    print('\n'.join(inspect.getsource(cartesia.TTS).splitlines()[:120]))
except Exception as e:
    print('tts_source_error', e)
if hasattr(cartesia, 'SynthesizeStream'):
    cls = cartesia.SynthesizeStream
    print('SynthesizeStream dir:', [x for x in dir(cls) if not x.startswith('_')])
    try:
        print('\nSource:\n')
        print(inspect.getsource(cls))
    except Exception as e:
        print('source_error', e)
else:
    print('no SynthesizeStream symbol')
if hasattr(cartesia, 'ChunkedStream'):
    cls = cartesia.ChunkedStream
    print('\nChunkedStream dir:', [x for x in dir(cls) if not x.startswith('_')])
    try:
        print('\nChunkedStream source:\n')
        print(inspect.getsource(cls))
    except Exception as e:
        print('chunk_source_error', e)
    try:
        import livekit.plugins.cartesia.tts as _tts_mod
        print('\nBase tts.ChunkedStream attrs:', [x for x in dir(_tts_mod.ChunkedStream) if not x.startswith('_')])
        import inspect as _inspect
        print('\nChunkedStream base source snippet:\n')
        print('\n'.join(_inspect.getsource(_tts_mod.ChunkedStream).splitlines()[:120]))
    except Exception as e:
        print('inspect_base_error', e)
    try:
        import livekit.plugins.cartesia.tts as _tts_mod
        print('\nSynthesisEventType attrs:', [x for x in dir(_tts_mod.SynthesisEventType) if not x.startswith('_')])
        print('\nSynthesizedAudioEmitter attrs:', [x for x in dir(_tts_mod.SynthesizedAudioEmitter) if not x.startswith('_')])
    except Exception as e:
        print('emit_inspect_error', e)
    try:
        import inspect as _inspect
        print('\ncollect sig:', _inspect.signature(_tts_mod.ChunkedStream.collect))
        print('done sig:', _inspect.signature(_tts_mod.ChunkedStream.done))
        print('exception sig:', _inspect.signature(_tts_mod.ChunkedStream.exception))
    except Exception as e:
        print('sig_error', e)
