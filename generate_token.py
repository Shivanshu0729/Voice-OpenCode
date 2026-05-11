from dotenv import load_dotenv
import os

from livekit.api import AccessToken, VideoGrants

load_dotenv()

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

token = (
    AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    .with_identity("shivanshu")
    .with_name("shivanshu")
    .with_grants(
        VideoGrants(
            room_join=True,
            room="voice-opencode-test",
        )
    )
    .to_jwt()
)

print("\nTOKEN:\n")
print(token)