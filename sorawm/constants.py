'''
Memory requirements for sorawm.constants:

- 1GB of VRAM is required per chunk when processing HD 1080p video.
- Processing a full HD 1080p movie typically requires around 16-18GB of VRAM for 1 chunk.
- Using 2 chunks per GB increases VRAM usage to approximately 22-26GB.
- Using 5 chunks per GB increases VRAM usage to approximately 26-36GB.

'''
CHUNK_SIZE_PER_GB_VRAM = 1 

