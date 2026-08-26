"""Fetch the Cityscapes data this project needs. Safe to rerun; it resumes.

  python download_cityscapes.py                 # labels, timestamps, train+val frames (~245 GB)
  python download_cityscapes.py --splits val    # val frames only (~35 GB)

Frames are extracted straight out of the remote 324 GB zip with parallel range
requests, so the zip never lands on disk. Login once; cookies are cached.
"""

import argparse
import getpass
import json
import os
import struct
import subprocess
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://www.cityscapes-dataset.com"
DATASET_ROOT = Path(os.environ["CITYSCAPES_ROOT"])
CACHE = DATASET_ROOT / ".cache"
SMALL = ["gtFine_trainvaltest.zip", "timestamp_sequence.zip"]  # downloaded whole, then extracted
SEQ = "leftImg8bit_sequence_trainvaltest.zip"  # 324 GB; extracted remotely, never downloaded whole


def login():
    s = requests.Session()
    cookies = CACHE / "cookies.json"
    if cookies.exists():
        s.cookies.update(json.loads(cookies.read_text()))

    r = s.get(BASE_URL + "/downloads/?list", allow_redirects=False)  # 200 only while the session is valid
    if r.status_code != 200:
        s.cookies.clear()
        user = os.environ.get("CITYSCAPES_USER") or input("Cityscapes username: ")
        pw = os.environ.get("CITYSCAPES_PASS") or getpass.getpass("Cityscapes password: ")

        s.get(BASE_URL + "/login", allow_redirects=False)  # sets the session cookie the POST needs
        data = {"username": user, "password": pw, "submit": "Login"}
        # Note: a successful login answers with a 302 redirect; anything else means bad credentials.
        if s.post(BASE_URL + "/login", data=data, allow_redirects=False).status_code != 302:
            raise SystemExit("Login failed: bad username or password.")

        cookies.write_text(json.dumps(s.cookies.get_dict()))
        cookies.chmod(0o600)  # owner-only: it holds a live session
        r = s.get(BASE_URL + "/downloads/?list", allow_redirects=False)

    return s, {p["name"]: p["packageID"] for p in r.json()}


def get_range(s, url, a, b, tries=8):
    for i in range(tries):
        try:
            r = s.get(
                url,
                headers={"Range": f"bytes={a}-{b}"},
                allow_redirects=False,
                timeout=(20, 180),  # (connect, read) seconds
            )

            # Note: check the length too; a dropped connection can still return a short 206 body.
            if r.status_code == 206 and len(r.content) == b - a + 1:
                return r.content
            if r.status_code in (301, 302, 401, 403):  # bounced to login: session expired
                raise SystemExit("Session expired. Rerun; it resumes.")
        except requests.RequestException:
            pass

        time.sleep(min(60, 2**i))  # exponential backoff, capped at 60 s

    raise OSError(f"failed bytes {a}-{b}")


def small_packages(s, ids):
    for name in SMALL:
        mark, z = CACHE / f"{name}.done", DATASET_ROOT / name
        if mark.exists():
            continue

        if not z.exists():
            part = z.with_suffix(".part")  # renamed to .zip only once fully downloaded
            with (
                s.get(f"{BASE_URL}/file-handling/?packageID={ids[name]}", stream=True) as r,
                part.open("wb") as f,
            ):
                r.raise_for_status()
                f.writelines(tqdm(r.iter_content(1 << 20), desc=name, unit="MB"))
            part.replace(z)

        zipfile.ZipFile(z).extractall(DATASET_ROOT)
        z.unlink()
        mark.touch()

    mark = CACHE / "trainIds.done"
    if not mark.exists():
        print("Creating train-ID label maps")
        # Note: cityscapesScripts locates the dataset through CITYSCAPES_DATASET, not CITYSCAPES_ROOT.
        subprocess.run(
            ["csCreateTrainIdLabelImgs"],
            check=True,
            env={**os.environ, "CITYSCAPES_DATASET": str(DATASET_ROOT)},
        )
        mark.touch()


def read_index(s, url):
    """[name, header_offset, compress_size, file_size, compress_type, CRC] per file, cached."""
    index = CACHE / "sequence_index.json"
    if not index.exists():
        print("Reading zip index (one-time)")
        size = int(s.get(url, headers={"Range": "bytes=0-0"}).headers["Content-Range"].split("/")[1])  # "0-0/<total>"

        # Note: zipfile only needs seek/tell/read; it reads the whole central directory in one read() call,
        # so building the index costs a handful of range requests.
        class Remote:
            pos = 0

            def tell(self):
                return self.pos

            def seek(self, o, whence=0):
                self.pos = (o, self.pos + o, size + o)[whence]  # whence: 0=start, 1=current, 2=end

            def read(self, n=-1):
                n = size - self.pos if n < 0 else min(n, size - self.pos)  # n=-1 reads to the end
                data = get_range(s, url, self.pos, self.pos + n - 1) if n > 0 else b""
                self.pos += n
                return data

        infos = [i for i in zipfile.ZipFile(Remote()).infolist() if not i.is_dir()]
        rows = [
            [
                i.filename,
                i.header_offset,
                i.compress_size,
                i.file_size,
                i.compress_type,
                i.CRC,
            ]
            for i in infos
        ]
        index.write_text(json.dumps(rows))

    return json.loads(index.read_text())


def span_end(e):
    # Note: the local header's extra-field length is only known after fetching it, so allow 1 KB of slack
    # (ZIP64 extras are ~20 bytes): 30-byte fixed header + name + extra field + compressed data.
    return e[1] + 30 + len(e[0]) + 1024 + e[2]


def write_member(e, blob, rel):
    name, _, csize, usize, ctype, crc = e
    n, m = struct.unpack_from("<HH", blob, rel + 26)  # local header: name length, extra-field length
    data = blob[rel + 30 + n + m : rel + 30 + n + m + csize]

    if ctype == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(data)  # -15: raw deflate, no zlib header
    if zlib.crc32(data) != crc:
        raise OSError(f"CRC mismatch: {name}")

    dest = DATASET_ROOT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Note: write to a temp file and rename, so an interrupted run never leaves a truncated frame
    # that the size check could mistake for a finished one.
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)

    return usize


def sequence(s, ids, splits, workers):
    url = f"{BASE_URL}/file-handling/?packageID={ids[SEQ]}"
    # Note: requests keeps only 10 pooled connections by default; with more threads the extras are discarded
    # and every request opens a new TLS connection, so size the pool to the worker count.
    s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers))

    prefixes = tuple(f"leftImg8bit_sequence/{sp}/" for sp in splits)  # also skips top-level README/license

    def done(e):
        p = DATASET_ROOT / e[0]
        return p.exists() and p.stat().st_size == e[3]  # present with the full uncompressed size

    todo = [e for e in read_index(s, url) if e[0].startswith(prefixes) and not done(e)]
    total = sum(e[3] for e in todo)
    print(f"{len(todo)} frames to fetch ({total / 1e9:.1f} GB)")

    # Note: the server caps throughput per connection, so fetch adjacent members together in ~32 MB range
    # requests; a gap over 64 KB (a skipped or already-done file) starts a new batch.
    batches = []
    for e in sorted(todo, key=lambda e: e[1]):  # by offset in the zip
        prev = batches[-1] if batches else None
        if prev and e[1] - prev[0][1] < 32 << 20 and e[1] < span_end(prev[-1]) + (64 << 10):
            prev.append(e)
        else:
            batches.append([e])

    def run(batch):
        a = batch[0][1]
        blob = get_range(s, url, a, span_end(batch[-1]))
        return sum(write_member(e, blob, e[1] - a) for e in batch)  # bytes written, for the progress bar

    failed = 0
    with (
        tqdm(total=total, unit="B", unit_scale=True, desc="frames") as pbar,
        ThreadPoolExecutor(workers) as ex,
    ):
        for f in as_completed([ex.submit(run, b) for b in batches]):
            try:
                pbar.update(f.result())
            except OSError as err:  # failed fetch or CRC; SystemExit (session expired) still aborts
                failed += 1
                print(err)

    if failed:
        raise SystemExit(f"{failed} batches failed. Rerun to fetch the rest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)  # holds cookies, zip index and .done markers
    s, ids = login()  # session + package name -> ID map
    small_packages(s, ids)  # gtFine, timestamps, train-ID label maps
    sequence(s, ids, args.splits.split(","), args.workers)  # sequence frames from the remote zip
