import inspect
import livekit.agents.pipeline.human_input as hi
src = inspect.getsource(hi)
print('\n'.join(src.splitlines()[:120]))
