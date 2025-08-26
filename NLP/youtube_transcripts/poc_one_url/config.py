# from pathlib import Path
# from urllib.parse import urlparse, parse_qs
# import re

# # Base dir = folder where these scripts live
# BASE_DIR = Path(__file__).resolve().parent
# OUTPUT_DIR = (BASE_DIR / "downloads")
# OUTPUT_DIR.mkdir(exist_ok=True)

# # Your video URL
# VIDEO_URL = "https://www.youtube.com/watch?v=guQe5ptefec"

# def extract_video_id(url: str) -> str:
#     parsed = urlparse(url)
#     v = parse_qs(parsed.query).get("v")
#     if v and len(v[0]) == 11:
#         return v[0]
#     if parsed.netloc.endswith("youtu.be"):
#         vid = parsed.path.strip("/").split("/")[0]
#         if len(vid) == 11:
#             return vid
#     m = re.search(r"(?:/embed/|/shorts/|v=)([\w-]{11})", url)
#     if m:
#         return m.group(1)
#     raise ValueError("Could not extract video id")
