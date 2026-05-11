from livekit.agents.utils import aio
print('Chan methods:', [x for x in dir(aio.Chan) if not x.startswith('_')])
