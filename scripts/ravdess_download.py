"""Read selected official ZIP members using validated HTTP byte ranges.

ZIP CRC validates extracted members; whole-archive MD5 is NOT verified in
range mode. Cached range blocks allow interrupted downloads to continue.
"""

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import subprocess
import time

SIZES = {"01": 552968353, "02": 570673024}


class RemoteArchive(io.RawIOBase):
    def __init__(self, actor, cache):
        self.size = SIZES[actor]
        self.position = 0
        self.url = f"https://zenodo.org/records/1188976/files/Video_Speech_Actor_{actor}.zip?download=1"
        self.cache = Path(cache) / f"ranges_actor_{actor}"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.prefix = Path(cache) / f"Video_Speech_Actor_{actor}.zip.part"
        self.tail = self.fetch(self.size - 65536, self.size - 1)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def fetch(self, start, end):
        path = self.cache / f"{start}-{end}.bin"
        length = end - start + 1
        if path.is_file() and path.stat().st_size == length:
            return path.read_bytes()
        temporary = path.with_suffix(".part")
        headers = path.with_suffix(".headers")
        for attempt in range(3):
            try:
                subprocess.run([
                    "curl.exe", "--location", "--fail", "--silent", "--show-error",
                    "--connect-timeout", "15", "--max-time", "40",
                    "--range", f"{start}-{end}", "--dump-header", str(headers),
                    "--output", str(temporary), self.url,
                ], check=True, timeout=50, capture_output=True)
                header_text = headers.read_text(encoding="latin-1").lower()
                expected = f"content-range: bytes {start}-{end}/{self.size}"
                if expected not in header_text or temporary.stat().st_size != length:
                    raise ValueError(f"Unexpected range response: {start}-{end}")
                temporary.replace(path)
                return path.read_bytes()
            except (subprocess.SubprocessError, ValueError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"Range download failed: {start}-{end}") from exc
                time.sleep(2 * (attempt + 1))

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.position
        n = min(n, self.size - self.position)
        if n <= 0:
            return b""
        start = self.position
        self.position += n
        if start >= self.size - len(self.tail):
            return self.tail[start - (self.size - len(self.tail)):start - (self.size - len(self.tail)) + n]
        if self.prefix.is_file() and start + n <= self.prefix.stat().st_size:
            with self.prefix.open("rb") as handle:
                handle.seek(start)
                return handle.read(n)
        # Bounded blocks avoid stalled full-archive transfers.
        chunks = [(pos, min(pos + 512 * 1024, start + n) - 1)
                  for pos in range(start, start + n, 512 * 1024)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            parts = list(executor.map(lambda bounds: self.fetch(*bounds), chunks))
        return b"".join(parts)
