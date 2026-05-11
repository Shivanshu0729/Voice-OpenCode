import inspect
import livekit.agents.pipeline.human_input as hi
src = inspect.getsource(hi)
start = src.find('def _recognize_task')
print(src[start:start+2400])
