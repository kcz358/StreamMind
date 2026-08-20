import io
import os
import struct
import sys
from elftools.elf.elffile import ELFFile

CORE = "/data/v-kaichen/azure_blob/b200_node/data/MatchTime/core/core"


def parse_note(n_type, data):
    info = {}
    if n_type == 0x3:  # NT_PRPSINFO
        # elf_prpsinfo: pr_state(1) sname(1) zomb(1) nice(1) pad flag(8) uid(4) gid(4) pid(4) ppid(4) pgrp(4) sid(4)
        # then pr_fname[16] and pr_psargs[80]
        try:
            fname = data[40:56].rstrip(b"\x00").decode("latin1", errors="replace")
            psargs = data[56:136].rstrip(b"\x00").decode("latin1", errors="replace")
            info["fname"] = fname
            info["psargs"] = psargs
        except Exception as e:
            info["err_prpsinfo"] = repr(e)
    elif n_type == 0x1:  # NT_PRSTATUS
        try:
            si_signo, si_code, si_errno = struct.unpack_from("<iii", data, 0)
            info["si_signo"] = si_signo
            info["si_code"] = si_code
            info["si_errno"] = si_errno
            pr_pid = struct.unpack_from("<i", data, 32)[0]
            info["pid"] = pr_pid
        except Exception as e:
            info["err_prstatus"] = repr(e)
    elif n_type == 0x53494749:  # NT_SIGINFO 'SIGI'
        try:
            si_signo, si_errno, si_code = struct.unpack_from("<iii", data, 0)
            info["signo"] = si_signo
            info["code"] = si_code
            info["errno"] = si_errno
        except Exception as e:
            info["err_siginfo"] = repr(e)
    elif n_type == 0x46494c45:  # NT_FILE 'FILE'
        try:
            count = struct.unpack_from("<Q", data, 0)[0]
            page_size = struct.unpack_from("<Q", data, 8)[0]
            info["nt_file_count"] = count
            info["page_size"] = page_size
            # after headers (16 + count*24), string table
            strings = data[16 + count * 24:]
            names = strings.split(b"\x00")
            info["files_sample"] = [n.decode("latin1", errors="replace") for n in names[:30] if n]
        except Exception as e:
            info["err_ntfile"] = repr(e)
    return info


def main():
    with open(CORE, "rb") as f:
        elf = ELFFile(f)
        print("ei_class:", elf.elfclass)
        print("machine:", elf["e_machine"])
        print("type:", elf["e_type"])
        print("n_segments:", elf.num_segments())
        note_seg = None
        for seg in elf.iter_segments():
            if seg["p_type"] == "PT_NOTE":
                note_seg = seg
                break
        if note_seg is None:
            print("no PT_NOTE"); return
        print("\n--- notes ---")
        name_to_type = {
            "NT_PRSTATUS": 0x1,
            "NT_PRPSINFO": 0x3,
            "NT_SIGINFO": 0x53494749,
            "NT_FILE": 0x46494c45,
        }
        for note in note_seg.iter_notes():
            raw = note["n_type"]
            n_type = raw if isinstance(raw, int) else name_to_type.get(raw, -1)
            name = note["n_name"]
            raw_desc = note.get("n_descdata", None)
            if raw_desc is None:
                raw_desc = note["n_desc"]
            if isinstance(raw_desc, bytes):
                desc = raw_desc
            elif isinstance(raw_desc, str):
                desc = raw_desc.encode("latin1", errors="ignore")
            else:
                desc = bytes(raw_desc)
            info = parse_note(n_type, desc)
            tag = raw if isinstance(raw, str) else f"0x{raw:x}"
            print(f"[{name}] type={tag} size={len(desc)}  {info}")


if __name__ == "__main__":
    main()
