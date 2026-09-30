#!/usr/bin/env python3
"""Replace bytes in a compiled ESPHome firmware and repair image integrity.

ESP-IDF / ESPHome app and bootloader images carry a 1-byte XOR checksum and,
usually, an appended SHA-256. Changing any payload byte without updating those
fields can make the image fail boot even though the flash write succeeded.

This script:
  1. Replaces every occurrence of a needle (text, hex, Base64 decoded to raw
     bytes — the HA API key case — or auto-detected UTF-8 vs Base64). The
     replacement may be the same length or shorter; leftover bytes are filled
     with a pad byte (default ``0x00``) so the image size and all offsets stay
     unchanged. A longer replacement is rejected because it would shift the
     firmware.
  2. Finds ESP-IDF images in a standalone ``firmware.bin`` or a merged
     ``firmware.factory.bin``.
  3. Rewrites each image's XOR checksum and SHA-256 so ESPHome / the ROM
     bootloader will accept the file.

Examples::

  python3 espbinpatch.py firmware.factory.bin \\
      --old-auto 'E1fyywUUE1DWzu0OzhDkyc4yAnfGwyEfsVNvhytrU6k=' \\
      --new-auto 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=' \\
      -o patched.factory.bin

  python3 espbinpatch.py firmware.factory.bin \\
      --old-b64 'E1fyywUUE1DWzu0OzhDkyc4yAnfGwyEfsVNvhytrU6k=' \\
      --new-b64 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=' \\
      -o patched.factory.bin

  python3 espbinpatch.py firmware.bin \\
      --old 'wifi_ssid' --new 'othername' -o patched.bin

  python3 espbinpatch.py firmware.bin \\
      --old 'very_long_ssid_name' --new 'short' -o patched.bin

  python3 espbinpatch.py firmware.factory.bin --verify

  python3 espbinpatch.py firmware.bin --find-placeholders
"""

import argparse
import base64
import binascii
import hashlib
import re
import struct
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

__version__ = "1.1.3"

IMAGE_MAGIC = 0xE9
CHECKSUM_INIT = 0xEF
HEADER_SIZE = 24
HASH_APPENDED_OFFSET = 23
MAX_SEGMENTS = 16
MAX_SEGMENT_LEN = 16 * 1024 * 1024

PARTITION_TABLE_OFFSETS = (0x8000, 0x9000)
PARTITION_ENTRY_SIZE = 32
PARTITION_MAGIC = 0x50AA
PARTITION_MAGIC_MD5 = 0xEBEB
PARTITION_TYPE_APP = 0x00

# Common places a bootloader image can start.
BOOTLOADER_CANDIDATES = (0x0, 0x1000)

# ASCII tokens like ESPBINPATCH_WIFI_SSID___________ (name, then optional pad underscores that set the needle's max length).
PLACEHOLDER_RE = re.compile(rb"ESPBINPATCH_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_*")


@dataclass(frozen=True)
class Segment:
    """One loadable segment: file offset of the payload and its length."""

    data_offset: int
    length: int


@dataclass
class EspImage:
    """Bounds and trailer of one ESP-IDF image inside a larger blob."""

    offset: int
    segment_count: int
    hash_appended: bool
    segments: tuple[Segment, ...]
    checksum_offset: int
    sha_offset: int | None
    end_offset: int

    @property
    def span(self) -> range:
        return range(self.offset, self.end_offset)


@dataclass(frozen=True)
class Replacement:
    """Decoded needle and replacement; ``encoding`` is set only for --old-auto."""

    old: bytes
    new: bytes
    encoding: str | None = None


class AutoEncodingError(Exception):
    """--old-auto could not uniquely choose UTF-8 vs Base64."""

    def __init__(self, message: str, *, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _u8(data: bytes, offset: int) -> int:
    return data[offset]


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def parse_esp_image(data: bytes, offset: int) -> EspImage | None:
    """Parse an ESP-IDF image at ``offset``, or return None if it is not valid."""
    if offset < 0 or offset + HEADER_SIZE > len(data):
        return None
    if _u8(data, offset) != IMAGE_MAGIC:
        return None

    segment_count = _u8(data, offset + 1)
    if not 1 <= segment_count <= MAX_SEGMENTS:
        return None

    hash_appended = _u8(data, offset + HASH_APPENDED_OFFSET) == 1
    cursor = offset + HEADER_SIZE
    segments: list[Segment] = []

    for _ in range(segment_count):
        if cursor + 8 > len(data):
            return None
        length = _u32(data, cursor + 4)
        if length == 0 or length > MAX_SEGMENT_LEN:
            return None
        data_offset = cursor + 8
        if data_offset + length > len(data):
            return None
        segments.append(Segment(data_offset=data_offset, length=length))
        cursor = data_offset + length

    # Checksum sits at the last byte of a 16-byte padded block.
    padding = (16 - 1) - (cursor % 16)
    checksum_offset = cursor + padding
    if checksum_offset >= len(data):
        return None

    sha_offset: int | None = None
    end_offset = checksum_offset + 1
    if hash_appended:
        sha_offset = end_offset
        end_offset += 32
        if end_offset > len(data):
            return None

    return EspImage(
        offset=offset,
        segment_count=segment_count,
        hash_appended=hash_appended,
        segments=tuple(segments),
        checksum_offset=checksum_offset,
        sha_offset=sha_offset,
        end_offset=end_offset,
    )


def xor_checksum(data: bytes, image: EspImage) -> int:
    state = CHECKSUM_INIT
    for segment in image.segments:
        for byte in data[segment.data_offset : segment.data_offset + segment.length]:
            state ^= byte
    return state & 0xFF


def image_digest(data: bytes, image: EspImage) -> bytes:
    return hashlib.sha256(data[image.offset : image.checksum_offset + 1]).digest()


def image_checksum_ok(data: bytes, image: EspImage) -> bool:
    if data[image.checksum_offset] != xor_checksum(data, image):
        return False
    if image.sha_offset is not None:
        if data[image.sha_offset : image.sha_offset + 32] != image_digest(data, image):
            return False
    return True


def repair_image(buf: bytearray, image: EspImage) -> None:
    buf[image.checksum_offset] = xor_checksum(buf, image)
    if image.sha_offset is not None:
        digest = image_digest(buf, image)
        buf[image.sha_offset : image.sha_offset + 32] = digest


def parse_partition_table(data: bytes, table_offset: int) -> list[tuple[int, int, str]]:
    """Return (offset, size, label) for APP partitions in a table, if valid."""
    if table_offset + PARTITION_ENTRY_SIZE > len(data):
        return []
    if _u16(data, table_offset) not in (PARTITION_MAGIC, PARTITION_MAGIC_MD5):
        return []

    apps: list[tuple[int, int, str]] = []
    for index in range(95):
        entry = table_offset + index * PARTITION_ENTRY_SIZE
        if entry + PARTITION_ENTRY_SIZE > len(data):
            break
        magic = _u16(data, entry)
        if magic == 0xFFFF or magic == 0x0000:
            break
        if magic == PARTITION_MAGIC_MD5:
            break
        if magic != PARTITION_MAGIC:
            return []
        part_type = _u8(data, entry + 2)
        offset = _u32(data, entry + 4)
        size = _u32(data, entry + 8)
        label = data[entry + 12 : entry + 28].split(b"\x00", 1)[0].decode("ascii", "replace")
        if part_type == PARTITION_TYPE_APP and size > 0:
            apps.append((offset, size, label or f"app@{offset:#x}"))
    return apps


def find_images(data: bytes) -> list[EspImage]:
    """Locate bootloader + app images in a factory blob or a lone app image."""
    found: dict[int, EspImage] = {}

    def consider(offset: int) -> None:
        if offset in found:
            return
        image = parse_esp_image(data, offset)
        if image is not None:
            found[offset] = image

    for offset in BOOTLOADER_CANDIDATES:
        consider(offset)

    for table_offset in PARTITION_TABLE_OFFSETS:
        for part_offset, _size, _label in parse_partition_table(data, table_offset):
            consider(part_offset)

    if not found:
        # Last resort: 4 KiB-aligned scan (covers unusual layouts).
        for offset in range(0, min(len(data), 2 * 1024 * 1024), 0x1000):
            consider(offset)

    return [found[key] for key in sorted(found)]


def padded_replacement(old: bytes, new: bytes, pad: int = 0x00) -> bytes:
    """Return ``new`` padded to ``len(old)`` so in-place firmware patches stay aligned."""
    if not old:
        raise ValueError("replacement needle must not be empty")
    if not 0 <= pad <= 255:
        raise ValueError(f"pad byte must be 0..255, got {pad}")
    if len(new) > len(old):
        raise ValueError(
            f"replacement must be at most as long as the needle "
            f"(old={len(old)} bytes, new={len(new)} bytes)"
        )
    return new + bytes([pad]) * (len(old) - len(new))


def find_offsets(data: bytes, needle: bytes) -> list[int]:
    """Return every non-overlapping start offset of ``needle`` in ``data``."""
    if not needle:
        return []
    hits: list[int] = []
    start = 0
    while True:
        index = data.find(needle, start)
        if index < 0:
            return hits
        hits.append(index)
        start = index + len(needle)


def replace_bytes(buf: bytearray, old: bytes, new: bytes, *, pad: int = 0x00) -> list[int]:
    """Overwrite every ``old`` span with ``new``, padding leftover bytes in place."""
    patched = padded_replacement(old, new, pad)
    hits: list[int] = []
    start = 0
    while True:
        index = buf.find(old, start)
        if index < 0:
            break
        buf[index : index + len(old)] = patched
        hits.append(index)
        start = index + len(old)
    return hits


def find_placeholders(data: bytes) -> list[tuple[int, bytes]]:
    """Return ``(offset, exact token)`` for each ``ESPBINPATCH_<name>`` in *data*.

    A name is required; trailing pad underscores are optional and included when
    present. They set the needle's maximum length (a replacement cannot be longer).
    """
    return [(match.start(), match.group()) for match in PLACEHOLDER_RE.finditer(data)]


def format_image(image: EspImage, data: bytes) -> str:
    status = "ok" if image_checksum_ok(data, image) else "INVALID"
    sha = f" sha256@{image.sha_offset:#x}" if image.sha_offset is not None else " no-sha256"
    return (
        f"  image @{image.offset:#x}  segments={image.segment_count}  "
        f"checksum@{image.checksum_offset:#x}{sha}  end={image.end_offset:#x}  [{status}]"
    )


def decode_hex(value: str) -> bytes:
    cleaned = "".join(value.split())
    try:
        return binascii.unhexlify(cleaned)
    except binascii.Error as exc:
        raise argparse.ArgumentTypeError(f"invalid hex: {exc}") from exc


def decode_b64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise argparse.ArgumentTypeError(f"invalid base64: {exc}") from exc


def decode_pad_byte(value: str) -> int:
    cleaned = value.strip().lower()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise argparse.ArgumentTypeError("pad byte is empty")
    try:
        number = int(cleaned, 16)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid pad byte (hex): {value}") from exc
    if not 0 <= number <= 255:
        raise argparse.ArgumentTypeError(f"pad byte must be 0x00..0xFF, got {value}")
    return number


def try_decode_candidates(value: str) -> list[tuple[str, bytes]]:
    """UTF-8 always; Base64 as well when the string is valid standard Base64."""
    utf8 = value.encode("utf-8")
    candidates: list[tuple[str, bytes]] = [("utf-8", utf8)]
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        return candidates
    if decoded and decoded != utf8:
        candidates.append(("base64", decoded))
    return candidates


def _describe_auto_encodings(value: str) -> str:
    parts = [f"utf-8 ({len(value.encode('utf-8'))} bytes)"]
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        parts.append("base64 (invalid)")
    else:
        parts.append(f"base64 ({len(decoded)} bytes)")
    return " or ".join(parts)


def _format_hit_offsets(hits: list[int], *, limit: int = 8) -> str:
    shown = ", ".join(f"{offset:#x}" for offset in hits[:limit])
    if len(hits) > limit:
        shown += f", … ({len(hits)} total)"
    return shown


def _decode_new_auto(encoding: str, new: str) -> bytes:
    """Decode ``new`` for --new-auto. Invalid Base64 is ``AutoEncodingError``."""
    if encoding == "utf-8":
        return new.encode("utf-8")
    if encoding == "base64":
        try:
            return decode_b64(new)
        except argparse.ArgumentTypeError as exc:
            raise AutoEncodingError(
                f"--new-auto is not valid base64: {exc}",
                exit_code=2,
            ) from exc
    raise AutoEncodingError(f"internal error: unknown encoding {encoding!r}", exit_code=2)


def resolve_auto_encoding(data: bytes, old: str, new: str) -> tuple[str, bytes, bytes]:
    """Pick UTF-8 or Base64 by which ``old`` needle exists in ``data``.

    ``new`` uses that same encoding when the replacement fits. If it is too
    long, the other encoding is used when it is valid and fits — so a 32-byte
    ASCII HA-key placeholder as Old can take a 44-character Base64 key as New.
    Invalid Base64 for ``new`` still errors when Old matched as Base64.
    """
    candidates = [(name, needle) for name, needle in try_decode_candidates(old) if needle]
    if not candidates:
        raise AutoEncodingError("replacement needle must not be empty", exit_code=2)

    matched: list[tuple[str, bytes, list[int]]] = []
    for name, needle in candidates:
        hits = find_offsets(data, needle)
        if hits:
            matched.append((name, needle, hits))

    if not matched:
        raise AutoEncodingError(
            f"needle not found as {_describe_auto_encodings(old)}",
            exit_code=1,
        )
    if len(matched) > 1:
        details = [
            f"{name} ({len(hits)} hit(s) at {_format_hit_offsets(hits)})"
            for name, _needle, hits in matched
        ]
        raise AutoEncodingError(
            "ambiguous encoding: found as "
            + " and ".join(details)
            + "\npass --old/--new for literal text, or --old-b64/--new-b64 for a decoded Base64 value",
            exit_code=2,
        )

    name, old_bytes, _hits = matched[0]
    new_bytes = _decode_new_auto(name, new)
    if len(new_bytes) <= len(old_bytes):
        return name, old_bytes, new_bytes

    other = "base64" if name == "utf-8" else "utf-8"
    try:
        alt = _decode_new_auto(other, new)
    except AutoEncodingError:
        return name, old_bytes, new_bytes
    if alt and len(alt) <= len(old_bytes):
        return f"{name}→{other}", old_bytes, alt
    return name, old_bytes, new_bytes


def build_test_image(payload: bytes, *, hash_appended: bool = True) -> bytes:
    """Minimal 1-segment ESP-IDF image used by --self-test."""
    header = bytearray(HEADER_SIZE)
    header[0] = IMAGE_MAGIC
    header[1] = 1
    header[HASH_APPENDED_OFFSET] = 1 if hash_appended else 0
    segment_header = struct.pack("<II", 0x3C000000, len(payload))
    body = bytes(header) + segment_header + payload
    padding = (16 - 1) - (len(body) % 16)
    checksum = CHECKSUM_INIT
    for byte in payload:
        checksum ^= byte
    image = bytearray(body + b"\x00" * padding + bytes([checksum & 0xFF]))
    if hash_appended:
        image.extend(hashlib.sha256(image).digest())
    return bytes(image)


def run_self_test() -> int:
    marker = b"HELLO-ESP-FIRMWARE-TEST!!"
    replacement = b"WORLD-ESP-FIRMWARE-TEST!!"
    assert len(marker) == len(replacement)

    payload = b"\x00" * 32 + marker + b"\xff" * 48
    original = bytearray(build_test_image(payload))
    image = parse_esp_image(original, 0)
    assert image is not None, "failed to parse synthetic image"
    assert image_checksum_ok(original, image), "synthetic image checksum broken"

    hits = replace_bytes(original, marker, replacement)
    assert hits == [HEADER_SIZE + 8 + 32], hits
    assert not image_checksum_ok(original, image), "expected checksum to break after patch"
    repair_image(original, image)
    assert image_checksum_ok(original, image), "repair failed"
    assert replacement in original
    assert marker not in original

    short_old = b"NEEDLE-TOO-LONG!!"
    short_new = b"TINY"
    short_padded = padded_replacement(short_old, short_new)
    assert short_padded == b"TINY" + b"\x00" * (len(short_old) - len(short_new))
    short_payload = b"\xaa" * 16 + short_old + b"\xbb" * 16
    short_buf = bytearray(build_test_image(short_payload))
    short_image = parse_esp_image(short_buf, 0)
    assert short_image is not None and image_checksum_ok(short_buf, short_image)
    short_hits = replace_bytes(short_buf, short_old, short_new)
    assert short_hits, "short needle not found"
    assert short_padded in short_buf
    assert short_old not in short_buf
    assert short_new in short_buf
    repair_image(short_buf, short_image)
    assert image_checksum_ok(short_buf, short_image), "repair failed after shorter patch"

    space_padded = padded_replacement(short_old, short_new, pad=0x20)
    assert space_padded == b"TINY" + b" " * (len(short_old) - len(short_new))
    zeroed = padded_replacement(short_old, b"")
    assert zeroed == b"\x00" * len(short_old)

    try:
        padded_replacement(b"abc", b"abcd")
    except ValueError:
        pass
    else:
        raise AssertionError("longer replacement must be rejected")

    # factory-style: bootloader-sized image + 0xFF gap + app at 0x10000
    boot = build_test_image(b"BOOT" + b"\x11" * 60)
    factory = bytearray(0x10000 + 4096)
    factory[:] = b"\xff" * len(factory)
    factory[0 : len(boot)] = boot
    app = build_test_image(b"xxxx" + marker + b"yyyy")
    factory[0x10000 : 0x10000 + len(app)] = app
    # fake partition table pointing at the app
    label = b"ota_0".ljust(16, b"\x00")
    entry = struct.pack("<HBBII", PARTITION_MAGIC, PARTITION_TYPE_APP, 0x10, 0x10000, 0x100000)
    entry += label + struct.pack("<I", 0)
    factory[0x8000 : 0x8000 + len(entry)] = entry
    factory[0x8000 + PARTITION_ENTRY_SIZE : 0x8000 + PARTITION_ENTRY_SIZE + 2] = b"\xff\xff"

    images = find_images(factory)
    assert any(img.offset == 0 for img in images), images
    assert any(img.offset == 0x10000 for img in images), images
    hits = replace_bytes(factory, marker, replacement)
    assert hits, "marker not found in factory blob"
    for img in images:
        if any(h in img.span for h in hits):
            repair_image(factory, img)
    app_image = parse_esp_image(factory, 0x10000)
    assert app_image is not None and image_checksum_ok(factory, app_image)

    key_raw = bytes(range(32))
    key_b64 = base64.b64encode(key_raw).decode("ascii")
    new_raw = bytes(32)
    new_b64 = base64.b64encode(new_raw).decode("ascii")
    ssid = "wifi_ssid"
    ssid_new = "othername"

    name, old_b, new_b = resolve_auto_encoding(key_raw, key_b64, new_b64)
    assert name == "base64", name
    assert old_b == key_raw
    assert new_b == new_raw

    name, old_b, new_b = resolve_auto_encoding(ssid.encode("utf-8"), ssid, ssid_new)
    assert name == "utf-8", name
    assert old_b == ssid.encode("utf-8")
    assert new_b == ssid_new.encode("utf-8")

    try:
        resolve_auto_encoding(b"\x00" * 64, ssid, ssid_new)
    except AutoEncodingError as exc:
        assert exc.exit_code == 1, exc.exit_code
        assert "utf-8" in str(exc) and "base64" in str(exc)
    else:
        raise AssertionError("missing needle must fail auto-detect")

    both = key_raw + key_b64.encode("ascii")
    try:
        resolve_auto_encoding(both, key_b64, new_b64)
    except AutoEncodingError as exc:
        assert exc.exit_code == 2, exc.exit_code
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("ambiguous encoding must fail auto-detect")

    try:
        resolve_auto_encoding(key_raw, key_b64, "!!!!not-b64!!!!")
    except AutoEncodingError as exc:
        assert exc.exit_code == 2, exc.exit_code
        assert "base64" in str(exc)
    else:
        raise AssertionError("invalid --new-auto must fail")

    # HA api.encryption.key placeholder: 32 ASCII bytes in the image, or the
    # YAML Base64 of those bytes as Old; New is the usual 44-char HA key.
    placeholder_ascii = "ESPBINPATCH_API_ENCRYPTION_KEY__"
    placeholder_raw = placeholder_ascii.encode("ascii")
    placeholder_b64 = base64.b64encode(placeholder_raw).decode("ascii")
    ha_new_b64 = "YcM9Kcc+DRDHfISFcF8FlDN0WuanM1LJV3duK/PugZA="
    ha_new_raw = base64.b64decode(ha_new_b64, validate=True)
    assert len(placeholder_raw) == 32 and len(ha_new_raw) == 32
    ha_payload = b"\x00" * 16 + placeholder_raw + b"\xff" * 16
    assert placeholder_b64.encode("ascii") not in ha_payload

    enc_ascii, old_ascii, new_ascii = resolve_auto_encoding(
        ha_payload, placeholder_ascii, ha_new_b64
    )
    assert enc_ascii == "utf-8→base64", enc_ascii
    assert old_ascii == placeholder_raw
    assert new_ascii == ha_new_raw

    enc_b64, old_from_b64, new_from_b64 = resolve_auto_encoding(
        ha_payload, placeholder_b64, ha_new_b64
    )
    assert enc_b64 == "base64", enc_b64
    assert old_from_b64 == old_ascii
    assert new_from_b64 == new_ascii

    patched_from_ascii = bytearray(ha_payload)
    replace_bytes(patched_from_ascii, old_ascii, new_ascii)
    patched_from_b64 = bytearray(ha_payload)
    replace_bytes(patched_from_b64, old_from_b64, new_from_b64)
    assert patched_from_ascii == patched_from_b64
    assert ha_new_raw in patched_from_ascii
    assert placeholder_raw not in patched_from_ascii

    # Same-encoding New already fits: do not switch to Base64.
    name, old_b, new_b = resolve_auto_encoding(b"abcdefghij", "abcdefghij", "YWJj")
    assert name == "utf-8", name
    assert new_b == b"YWJj"

    # Too-long New that is not Base64 stays utf-8 so padded_replacement rejects.
    name, old_b, new_b = resolve_auto_encoding(
        ha_payload, placeholder_ascii, "this-is-not-valid-base64-and-is-way-too-long!!"
    )
    assert name == "utf-8", name
    assert len(new_b) > len(old_b)

    ns = parse_args(["fw.bin", "--old", key_b64, "--new", key_b64])
    repl = resolve_replacement(ns, key_raw + key_b64.encode("ascii"))
    assert repl is not None
    assert repl.encoding is None
    assert repl.old == key_b64.encode("utf-8")
    assert repl.old != key_raw

    ns = parse_args(["fw.bin", "--old-b64", key_b64, "--new-b64", new_b64])
    repl = resolve_replacement(ns, both)
    assert repl is not None
    assert repl.encoding is None
    assert repl.old == key_raw
    assert repl.new == new_raw

    ns = parse_args(["fw.bin", "--old-auto", key_b64, "--new-auto", new_b64])
    repl = resolve_replacement(ns, key_raw)
    assert repl is not None
    assert repl.encoding == "base64"
    assert repl.old == key_raw
    assert repl.new == new_raw

    ns = parse_args(["fw.bin", "--old-auto", ssid, "--new-auto", ssid_new])
    repl = resolve_replacement(ns, ssid.encode("utf-8"))
    assert repl is not None
    assert repl.encoding == "utf-8"
    assert repl.old == ssid.encode("utf-8")

    wifi_ssid = b"ESPBINPATCH_WIFI_SSID___________"
    wifi_password = b"ESPBINPATCH_WIFI_PASSWORD______________________________________"
    ota_password = b"ESPBINPATCH_OTA_PASSWORD________________________________________"
    api_key = placeholder_raw
    blob = (
        b"\x00" * 8
        + wifi_ssid
        + b"\x11" * 4
        + wifi_password
        + b"noise ESPBINPATCH_ not-a-match"
        + b"\x00"
        + ota_password
        + b"\x00"
        + api_key
        + b"\x00ESPBINPATCH"
    )
    found = find_placeholders(blob)
    assert [token for _offset, token in found] == [
        wifi_ssid,
        wifi_password,
        ota_password,
        api_key,
    ], found
    assert [offset for offset, _token in found] == [
        blob.find(wifi_ssid),
        blob.find(wifi_password),
        blob.find(ota_password),
        blob.find(api_key),
    ]
    assert find_placeholders(b"ESPBINPATCH_") == []
    assert find_placeholders(b"no placeholders here") == []
    assert find_placeholders(b"ESPBINPATCH_WIFI_SSID___________noise") == [
        (0, wifi_ssid)
    ]

    auto_payload = b"\x00" * 16 + key_raw + ssid.encode("utf-8") + b"\xff" * 16
    image_bytes = build_test_image(auto_payload)
    ha_image = build_test_image(ha_payload)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        tmp.write_bytes(image_bytes)
        assert main([str(tmp), "--old-auto", key_b64, "--new-auto", new_b64, "--dry-run"]) == 0
        assert main([str(tmp), "--old-auto", ssid, "--new-auto", ssid_new, "--dry-run"]) == 0
        assert main([str(tmp), "--old", ssid, "--new", ssid_new, "--dry-run"]) == 0
        assert main([str(tmp), "--old-b64", key_b64, "--new-b64", new_b64, "--dry-run"]) == 0
        tmp.write_bytes(ha_image)
        assert main(
            [str(tmp), "--old-auto", placeholder_ascii, "--new-auto", ha_new_b64, "--dry-run"]
        ) == 0
        assert main(
            [str(tmp), "--old-auto", placeholder_b64, "--new-auto", ha_new_b64, "--dry-run"]
        ) == 0
        tmp.write_bytes(blob)
        stdout = StringIO()
        with redirect_stdout(stdout):
            assert main([str(tmp), "--find-placeholders"]) == 0
        assert stdout.getvalue().splitlines() == [
            wifi_ssid.decode("ascii"),
            wifi_password.decode("ascii"),
            ota_password.decode("ascii"),
            api_key.decode("ascii"),
        ]
        tmp.write_bytes(b"no ESPBINPATCH tokens")
        stderr = StringIO()
        with redirect_stdout(StringIO()), redirect_stderr(stderr):
            assert main([str(tmp), "--find-placeholders"]) == 1
        assert "no ESPBINPATCH placeholders found" in stderr.getvalue()
    finally:
        tmp.unlink(missing_ok=True)

    print("self-test passed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace bytes in an ESPHome firmware image and repair ESP-IDF checksums."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "firmware",
        nargs="?",
        type=Path,
        help="Input firmware image (.bin or .espbinpatch; firmware.bin or firmware.factory.bin)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Write patched image here")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input file")
    parser.add_argument("--old", help="UTF-8 text to find")
    parser.add_argument(
        "--new",
        help="UTF-8 replacement (same length or shorter than --old; leftover bytes are padded)",
    )
    parser.add_argument("--old-hex", help="Hex bytes to find (whitespace ignored)")
    parser.add_argument(
        "--new-hex",
        help="Hex replacement (same length or shorter than --old-hex; leftover bytes are padded)",
    )
    parser.add_argument(
        "--old-b64",
        help="Standard Base64 value; the decoded bytes are searched (HA API key)",
    )
    parser.add_argument(
        "--new-b64",
        help="Standard Base64 replacement; decoded length must be <= --old-b64",
    )
    parser.add_argument(
        "--old-auto",
        help="Find UTF-8 text or decoded Base64, whichever needle exists in the firmware",
    )
    parser.add_argument(
        "--new-auto",
        help="Replacement in the same encoding as --old-auto (decoded length must be <= needle)",
    )
    parser.add_argument(
        "--pad",
        default=0,
        type=decode_pad_byte,
        metavar="HEX",
        help="Hex pad byte for leftover space when --new is shorter (default: 00)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only parse images and report whether checksums/SHA-256 match",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be replaced; do not write a file",
    )
    parser.add_argument("--self-test", action="store_true", help="Run built-in integrity tests")
    parser.add_argument(
        "--find-placeholders",
        action="store_true",
        help=(
            "List ESPBINPATCH_<name> tokens as they appear in the firmware "
            "(trailing pad underscores optional, included if present, set the "
            "needle's max length); do not patch"
        ),
    )
    return parser.parse_args(argv)


def resolve_replacement(args: argparse.Namespace, data: bytes) -> Replacement | None:
    pairs = [
        ("--old/--new", args.old, args.new, lambda v: v.encode("utf-8")),
        ("--old-hex/--new-hex", args.old_hex, args.new_hex, decode_hex),
        ("--old-b64/--new-b64", args.old_b64, args.new_b64, decode_b64),
    ]
    chosen: list[Replacement] = []
    for name, old, new, decoder in pairs:
        if old is None and new is None:
            continue
        if old is None or new is None:
            raise SystemExit(f"{name} must be given together")
        chosen.append(Replacement(old=decoder(old), new=decoder(new)))

    if args.old_auto is not None or args.new_auto is not None:
        if args.old_auto is None or args.new_auto is None:
            raise SystemExit("--old-auto/--new-auto must be given together")
        encoding, old_bytes, new_bytes = resolve_auto_encoding(
            data, args.old_auto, args.new_auto
        )
        chosen.append(Replacement(old=old_bytes, new=new_bytes, encoding=encoding))

    if len(chosen) > 1:
        raise SystemExit("use only one of --old, --old-hex, --old-b64, or --old-auto")
    if not chosen:
        return None
    return chosen[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.self_test:
        return run_self_test()

    if args.firmware is None:
        print("firmware path is required (or pass --self-test)", file=sys.stderr)
        return 2

    data = bytearray(args.firmware.read_bytes())
    if args.find_placeholders:
        found = find_placeholders(data)
        if not found:
            print("no ESPBINPATCH placeholders found", file=sys.stderr)
            return 1
        for _offset, token in found:
            print(token.decode("ascii"))
        return 0

    images = find_images(data)
    if not images:
        print("no ESP-IDF images found (not an ESPHome .bin?)", file=sys.stderr)
        return 1

    print(f"{args.firmware}  {len(data)} bytes")
    for image in images:
        print(format_image(image, data))

    if args.verify:
        bad = [image for image in images if not image_checksum_ok(data, image)]
        return 1 if bad else 0

    try:
        replacement = resolve_replacement(args, data)
    except AutoEncodingError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    if replacement is None:
        print(
            "nothing to replace (pass --old/--new, --old-hex, --old-b64, or --old-auto)",
            file=sys.stderr,
        )
        return 2

    old, new = replacement.old, replacement.new
    if replacement.encoding is not None:
        print(f"encoding: {replacement.encoding} ({len(old)}-byte needle)")
    pad = args.pad
    try:
        hits = replace_bytes(data, old, new, pad=pad)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if len(new) < len(old):
        print(
            f"replaced {len(hits)} occurrence(s) of {len(old)} byte(s) "
            f"with {len(new)} byte(s), padded {len(old) - len(new)} byte(s) "
            f"with 0x{pad:02x}"
        )
    else:
        print(f"replaced {len(hits)} occurrence(s) of {len(old)} byte(s)")
    for hit in hits:
        print(f"  at {hit:#x}")
    if not hits:
        print("needle not found; file unchanged", file=sys.stderr)
        return 1

    for image in images:
        overlaps = [h for h in hits if any(h + i in image.span for i in range(len(old)))]
        if not overlaps and image_checksum_ok(data, image):
            continue
        before_ok = image_checksum_ok(data, image)
        repair_image(data, image)
        print(
            f"repaired image @{image.offset:#x}  "
            f"(checksum was {'ok' if before_ok else 'stale'} → "
            f"{'ok' if image_checksum_ok(data, image) else 'STILL BAD'})"
        )

    still_bad = [image for image in images if not image_checksum_ok(data, image)]
    if still_bad:
        print("failed to repair all images", file=sys.stderr)
        return 1

    if args.dry_run:
        print("dry-run: not writing a file")
        return 0

    if args.in_place:
        dest = args.firmware
    elif args.output:
        dest = args.output
    else:
        print("pass -o OUTPUT or --in-place to write the patched file", file=sys.stderr)
        return 2

    dest.write_bytes(data)
    print(f"wrote {dest}  ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
