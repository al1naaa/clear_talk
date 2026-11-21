"""Read public archive metadata using bounded HTTP ranges, without full-corpus downloads."""
import io
import urllib.request
import zipfile

KSD_URL = "https://openslr.elda.org/resources/140/audio2.zip"


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise OSError("Server must support partial downloads")
            self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative archive offset")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 32 * 1024 ** 2:
            raise OSError("Refusing unbounded archive read")
        start, end = self.position, self.position + size - 1
        key = (start, end)
        if key not in self.cache:
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                    raise OSError("Server ignored or changed the requested range")
                payload = response.read(size + 1)
                if len(payload) != size:
                    raise OSError("Incomplete archive range")
            self.cache[key] = payload
        self.position += size
        return self.cache[key]


def inventory():
    with zipfile.ZipFile(RangeReader(KSD_URL)) as archive:
        records = archive.infolist()
        print("Archive members:", len(records))
        print("Example paths:", [item.filename for item in records[:12]])
        text_files = [item for item in records if item.filename.endswith((".txt", ".csv", ".tsv"))]
        print("Transcript/metadata files:", [(item.filename, item.file_size) for item in text_files[:10]])
        if text_files and text_files[0].file_size < 200000:
            print(archive.read(text_files[0]).decode("utf-8-sig")[:1000])


if __name__ == "__main__":
    inventory()
