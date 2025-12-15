from enum import Enum


class CleanerType(str, Enum):
    LAMA = "lama"
    E2FGVI = "e2fgvi"
    E2FGVI_HQ = "e2fgvi_hq"