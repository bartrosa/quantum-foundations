"""Console entry points for publication-side reproducibility."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from quantum_foundations.conway_games.experiments import (
    e4_conway_aut_scaling,
    e5_outcome_distribution,
    e6_hierarchy_collapse,
    e7_conway_gibbs,
)
from quantum_foundations.entropic_causets.experiments import (
    e1_aut_scaling,
    e2_gibbs,
    e3_verlinde,
)
from quantum_foundations.entropic_causets.logging_setup import configure_main_process_logging
from quantum_foundations.sedenion.edge_fano import verify_edge_fano_bijection
from quantum_foundations.sedenion.probe_verification import verify_probe_classification


def _flush_logging_handlers() -> None:
    root = logging.getLogger("quantum_foundations")
    for handler in root.handlers:
        try:
            handler.flush()
        except Exception:
            continue


def main_verify_probe() -> None:
    result = verify_probe_classification()
    print(f"Configurations checked: {result.config_count} (expect 28)")
    print(f"Probe-class histogram: {result.class_histogram}")
    if result.mismatches:
        print("FAIL")
        for line in result.mismatches[:50]:
            print(line)
        if len(result.mismatches) > 50:
            print(f"... {len(result.mismatches)} total")
        sys.exit(1)
    print(result.canonical_table_tex)
    print(
        "All 28 configurations verified: classification matches analytic proof "
        "(Lemma 3.2 + Theorems 3.4, 3.6)."
    )


def main_verify_edge_fano() -> None:
    mismatches = verify_edge_fano_bijection()
    if mismatches:
        print("FAIL")
        for m in mismatches:
            print(m)
        sys.exit(1)
    print("All 28 bijection checks: PASS")


def _parse_e1_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E1 |Aut| scaling experiment.")
    parser.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("papers/entropic-causets/results"))
    return parser.parse_args()


def main_run_e1_aut_scaling() -> None:
    """qf-run-e1: |Aut| scaling experiment."""
    args = _parse_e1_args()
    log_path, listener, log_queue = configure_main_process_logging(
        "e1_aut_scaling",
        args.output_dir,
    )
    listener.start()
    log = logging.getLogger("quantum_foundations.cli.e1")
    log.info("starting E1 with args: %s", vars(args))
    ns = tuple(n for n in (20, 30, 50, 70, 100, 150, 200) if n <= args.max_n)
    try:
        result = e1_aut_scaling.run(
            n_workers=args.n_workers,
            seeds=args.seeds,
            ns=ns,
            log_queue=log_queue,
        )
        e1_aut_scaling.write_outputs(result, args.output_dir, n_workers=args.n_workers)
        f1 = result.fit.get("log_aut_pos")
        log.info(
            "E1 finished | n_records=%d %s | log=%s",
            len(result.rows),
            f"slope={f1.slope:.6f} r2={f1.r2:.6f}" if f1 else "fit=—",
            log_path,
        )
    except Exception:
        log.exception("E1 crashed")
        raise
    finally:
        listener.stop()


def _parse_e2_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2 Gibbs paradox test.")
    parser.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("papers/entropic-causets/results"))
    return parser.parse_args()


def main_run_e2_gibbs() -> None:
    """qf-run-e2: Gibbs paradox test."""
    args = _parse_e2_args()
    log_path, listener, log_queue = configure_main_process_logging("e2_gibbs", args.output_dir)
    listener.start()
    log = logging.getLogger("quantum_foundations.cli.e2")
    log.info("starting E2 with args: %s", vars(args))
    ns = tuple(n for n in (15, 20, 30) if n <= args.max_n)
    try:
        result = e2_gibbs.run(
            n_workers=args.n_workers,
            seeds=args.seeds,
            ns=ns,
            log_queue=log_queue,
        )
        e2_gibbs.write_outputs(result, args.output_dir)
        log.info(
            "E2 finished | pass=%d fail=%d max_deviation=%.6e | log=%s",
            result.summary.pass_count,
            result.summary.fail_count,
            result.summary.max_abs_deviation,
            log_path,
        )
        if result.summary.fail_count > 0 or result.summary.s_a_fail_count > 0:
            sys.exit(1)
    except Exception:
        log.exception("E2 crashed")
        raise
    finally:
        listener.stop()


def _parse_e3_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E3 Verlinde distance scaling.")
    parser.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("papers/entropic-causets/results"))
    return parser.parse_args()


def main_run_e3_verlinde() -> None:
    """qf-run-e3: Verlinde distance scaling."""
    args = _parse_e3_args()
    log_path, listener, log_queue = configure_main_process_logging("e3_verlinde", args.output_dir)
    listener.start()
    log = logging.getLogger("quantum_foundations.cli.e3")
    log.info("starting E3 with args: %s", vars(args))
    try:
        result = e3_verlinde.run(n_workers=args.n_workers, seeds=args.seeds, log_queue=log_queue)
        e3_verlinde.write_outputs(result, args.output_dir)
        f3 = result.fit.get("delta_s_b_pos")
        log.info(
            "E3 finished | %s interpretation=%s | log=%s",
            f"ΔS_B pos slope={f3.slope:.6f} r2={f3.r2:.6f}" if f3 else "fit=—",
            result.interpretation,
            log_path,
        )
    except Exception:
        log.exception("E3 crashed")
        raise
    finally:
        listener.stop()


def _parse_e4_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E4 Conway |Aut| scaling.")
    parser.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("papers/conway-causets/results"))
    return parser.parse_args()


def main_run_e4_conway_aut_scaling() -> None:
    args = _parse_e4_args()
    log_path, listener, log_queue = configure_main_process_logging(
        "e4_conway_aut_scaling",
        args.output_dir,
    )
    listener.start()
    log = logging.getLogger("quantum_foundations.cli.e4")
    ns = tuple(n for n in (15, 20, 30, 50, 70, 100) if n <= args.max_n)
    interrupted = False
    try:
        result = e4_conway_aut_scaling.run(
            n_workers=args.n_workers,
            seeds=args.seeds,
            ns=ns,
            log_queue=log_queue,
        )
        e4_conway_aut_scaling.write_outputs(result, args.output_dir)
        log.info("E4 finished | n_rows=%d | log=%s", len(result.rows), log_path)
    except KeyboardInterrupt:
        interrupted = True
        log.warning("CLI caught KeyboardInterrupt that escaped run()")
    finally:
        try:
            listener.stop()
        except Exception:
            pass
    if interrupted:
        _flush_logging_handlers()
        os._exit(130)


def _parse_e5_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E5 outcome distribution.")
    parser.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("papers/conway-causets/results"))
    return parser.parse_args()


def main_run_e5_outcome_distribution() -> None:
    args = _parse_e5_args()
    log_path, listener, log_queue = configure_main_process_logging(
        "e5_outcome_distribution",
        args.output_dir,
    )
    listener.start()
    log = logging.getLogger("quantum_foundations.cli.e5")
    ns = tuple(n for n in (30, 50, 100, 200) if n <= args.max_n)
    interrupted = False
    try:
        result = e5_outcome_distribution.run(
            n_workers=args.n_workers,
            seeds=args.seeds,
            ns=ns,
            log_queue=log_queue,
        )
        e5_outcome_distribution.write_outputs(result, args.output_dir)
        log.info("E5 finished | n_rows=%d | log=%s", len(result.rows), log_path)
    except KeyboardInterrupt:
        interrupted = True
        log.warning("CLI caught KeyboardInterrupt that escaped run()")
    finally:
        try:
            listener.stop()
        except Exception:
            pass
    if interrupted:
        _flush_logging_handlers()
        os._exit(130)


def _parse_e6_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E6 hierarchy collapse.")
    parser.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("papers/conway-causets/results"))
    return parser.parse_args()


def main_run_e6_hierarchy_collapse() -> None:
    args = _parse_e6_args()
    log_path, listener, log_queue = configure_main_process_logging(
        "e6_hierarchy_collapse",
        args.output_dir,
    )
    listener.start()
    log = logging.getLogger("quantum_foundations.cli.e6")
    ns = tuple(n for n in (20, 30, 50, 70, 100) if n <= args.max_n)
    interrupted = False
    try:
        result = e6_hierarchy_collapse.run(
            n_workers=args.n_workers,
            seeds=args.seeds,
            ns=ns,
            log_queue=log_queue,
        )
        e6_hierarchy_collapse.write_outputs(result, args.output_dir)
        log.info("E6 finished | n_rows=%d | log=%s", len(result.rows), log_path)
    except KeyboardInterrupt:
        interrupted = True
        log.warning("CLI caught KeyboardInterrupt that escaped run()")
    finally:
        try:
            listener.stop()
        except Exception:
            pass
    if interrupted:
        _flush_logging_handlers()
        os._exit(130)


def _parse_e7_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E7 Conway Gibbs checks.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("papers/conway-causets/results"))
    return parser.parse_args()


def main_run_e7_conway_gibbs() -> None:
    args = _parse_e7_args()
    log_path, listener, _log_queue = configure_main_process_logging(
        "e7_conway_gibbs",
        args.output_dir,
    )
    listener.start()
    log = logging.getLogger("quantum_foundations.cli.e7")
    ns = tuple(n for n in (10, 15, 20) if n <= args.max_n)
    try:
        result = e7_conway_gibbs.run(seeds=args.seeds, ns=ns)
        e7_conway_gibbs.write_outputs(result, args.output_dir)
        if result.summary.fail_count > 0 or result.summary.s_a_fail_count > 0:
            sys.exit(1)
        log.info(
            "E7 finished | pass=%d fail=%d | log=%s",
            result.summary.pass_count,
            result.summary.fail_count,
            log_path,
        )
    finally:
        listener.stop()
