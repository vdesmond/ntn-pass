"""ntn-pass: locating a ground transmitter from satellite delay and Doppler, for NR NTN and COSPAS-SARSAT."""

import argparse
import importlib

# In dependency order: each study uses what the ones before it established.
STUDIES = {
    "passes": "pass geometry, serving timeline and handover rates",
    "link": "CNR and C/N0 along passes (TR 38.821 parameter sets)",
    "timing": "NR NTN uplink timing: TA, SIB19 validity, K_offset, HARQ",
    "acquisition": "SSB acquisition under Doppler and per-burst delay / Doppler accuracy",
    "beacon": "406 MHz beacon link through Galileo to a MEOLUT, TOA / FOA bounds",
    "locate": "locating the transmitter: NR location verification, MEOSAR, LEOSAR",
}


def main() -> None:
    ap = argparse.ArgumentParser(prog="ntn-pass")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in STUDIES.items():
        p = sub.add_parser(name, help=help_)
        if name == "passes":
            p.add_argument("--site", default="sophia", choices=["sophia", "toulouse"])
    sub.add_parser("all", help="every study in dependency order (~5 min)")
    args = ap.parse_args()
    if args.cmd != "all":
        kw = {"site_key": args.site} if args.cmd == "passes" else {}
        importlib.import_module(f"ntn_pass.studies.{args.cmd}").run(**kw)
        return
    for name in STUDIES:
        print(f"== {name}")
        importlib.import_module(f"ntn_pass.studies.{name}").run()
