import click
import functools
from logging import getLogger
import os
import sys
import re
import yaml
import json
import csv
import time
import subprocess
from decimal import Decimal
from pathlib import Path
from dataclasses import dataclass
from jinja2 import Template
from .version import VERSION

_log = getLogger(__name__)


@click.version_option(version=VERSION, prog_name="benchmgr")
@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def set_verbose(verbose: bool | None):
    from logging import basicConfig

    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    level = "INFO"
    if verbose:
        level = "DEBUG"
    elif verbose is not None:
        level = "WARNING"
    basicConfig(level=level, format=fmt)


def verbose_option(func):
    @click.option("--verbose/--quiet", default=None, help="log level")
    @functools.wraps(func)
    def _(verbose, *args, **kwargs):
        set_verbose(verbose)
        return func(*args, **kwargs)

    return _


def represent_decimal(dumper: yaml.representer.Representer, instance: Decimal):
    return dumper.represent_float(float(instance))


def decimal_to_float(obj: Decimal):
    if isinstance(obj, Decimal):
        return float(obj)


def format_option(default="json"):
    def _wrap(func):
        @click.option("--format", default=default, type=click.Choice(["yaml", "json", "csv"]), show_default=True)
        @functools.wraps(func)
        def _(format, *args, **kwargs):
            res = func(*args, **kwargs)
            if format == "yaml":
                yaml.add_representer(Decimal, represent_decimal)
                yaml.dump(res, sys.stdout)
            elif format == "json":
                json.dump(res, sys.stdout, default=decimal_to_float)
            elif format == "csv":
                names = {}
                for v in res:
                    names.update(v)
                wr = csv.DictWriter(sys.stdout, fieldnames=names.keys())
                wr.writeheader()
                wr.writerows(res)

        return _

    return _wrap


def config_option(func):
    @click.option("--config", type=click.File(), required=True, envvar="BENCHMGR_CONFIG", show_envvar=True)
    @click.option("--mode", required=True, default="wrk", envvar="BENCHMGR_MODE", show_default=True, show_envvar=True)
    @functools.wraps(func)
    def _(config, mode, *args, **kwargs):
        confdata = yaml.safe_load(config)
        for i in confdata:
            if i.get("name") == mode:
                return func(config=i, *args, **kwargs)
        modes = [x.get("name") for x in confdata if "name" in x]
        raise click.BadArgumentUsage(f"invalid mode: {modes}")

    return _


def parse_time(s: str) -> Decimal:
    suffixmap = {
        "us": Decimal(1) / Decimal(1000000),
        "μs": Decimal(1) / Decimal(1000000),
        "ms": Decimal(1) / Decimal(1000),
        "ns": Decimal(1) / Decimal(1000000000),
        "s": Decimal(1),
        "m": Decimal(60),
        "h": Decimal(60 * 60),
    }
    for k, v in suffixmap.items():
        if s.endswith(k):
            s = s.removesuffix(k)
            return Decimal(s) * v
    return Decimal(s)


class TimeParam(click.ParamType):
    name = "time"

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            return parse_time(value)
        return value


def parse_num(s: str) -> Decimal:
    suffixmap = {
        "M": Decimal(1000000),
        "m": Decimal(1000000),
        "K": Decimal(1000),
        "k": Decimal(1000),
    }
    for k, v in suffixmap.items():
        if s.endswith(k):
            s = s.removesuffix(k)
            return Decimal(s) * v
    return Decimal(s)


def parse1(config, file: Path) -> dict[str, int | Decimal]:
    intkey = {"concurrency", "threads", "connections", "fail"}
    res: dict[str, int | Decimal] = {}
    with open(file) as ifp:
        for line in ifp.readlines():
            for reg in config.get("result"):
                m = re.search(reg, line)
                if m:
                    match = m.groupdict()
                    if "percentile" in match:
                        key: str = match.pop("percentile") + "%tile"
                        if "percentile_sec" in match:
                            res[key] = Decimal(match.pop("percentile_sec"))
                        elif "percentile_ms" in match:
                            res[key] = Decimal(match.pop("percentile_ms")) / Decimal(1000)
                        elif "percentile_us" in match:
                            res[key] = Decimal(match.pop("percentile_us")) / Decimal(1000000)
                        elif "percentile_time" in match:
                            res[key] = parse_time(match.pop("percentile_time"))
                    for k, v in match.items():
                        if k.endswith("_time"):
                            k = k.removesuffix("_time")
                            v = parse_time(v)
                        elif k.endswith("_num"):
                            k = k.removesuffix("_num")
                            v = parse_num(v)
                        _log.debug("k=%s, val=%s", k, v)
                        if k in intkey:
                            v = int(v)
                        else:
                            v = Decimal(v)
                        if k.endswith("_ms"):
                            k = k.removesuffix("_ms")
                            v = v / Decimal(1000)
                        elif k.endswith("_sec"):
                            k = k.removesuffix("_sec")
                            # val = val
                        elif k.endswith("_us"):
                            k = k.removesuffix("_us")
                            v = v / Decimal(1000000)
                        res[k] = v
    return res


@cli.command()
@verbose_option
@config_option
@format_option()
@click.argument("datafiles", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def parse(config, datafiles: tuple[str]):
    """parse result files"""
    return [dict(filename=x, **parse1(config, Path(x))) for x in datafiles]


@dataclass
class Cparam:
    concurrency: int
    latency: Decimal = Decimal(0)
    throughput: Decimal = Decimal(0)


def select_chk(results: list[Cparam], suggest: Cparam) -> bool:
    # check dup
    if any([x.concurrency == suggest.concurrency for x in results]):
        return False
    return True


def select_cparam(results: list[Cparam], limit: Decimal) -> Cparam | None:
    if not results:
        return Cparam(1)
    prevr = sorted(results, key=lambda f: f.concurrency)
    while True:
        if len(prevr) <= 2:
            break
        if prevr[-2].latency < limit:
            break
        prevr.pop()
    prevc = sorted(prevr, key=lambda f: f.concurrency)
    if prevc[-1].latency < limit and (
        prevc[-1].latency == 0 or prevc[-1].concurrency / prevc[-1].latency < prevc[-1].throughput * Decimal(1.5)
    ):
        try:
            suggest = int(prevc[-1].concurrency * (limit / prevc[-1].latency))
            _log.debug("choose ext1 %s", suggest)
        except ZeroDivisionError:
            suggest = prevc[-1].concurrency + 10
            _log.debug("choose ext0 %s", suggest)
        if prevc[-1].concurrency * 3 < suggest:
            suggest = int(prevc[-1].concurrency * 3)
            _log.debug("choose ext100 %s", suggest)
        return Cparam(suggest)
    # select max distance(throughput)
    prevt = sorted(prevr, key=lambda f: f.throughput)
    diff = [(x, prevt[x + 1].throughput - prevt[x].throughput) for x in range(len(prevt) - 1)]
    diff.sort(key=lambda f: f[-1])
    idx = diff[-1][0]
    suggest = Cparam(int((prevt[idx].concurrency + prevt[idx + 1].concurrency) / 2))
    if select_chk(results, suggest):
        _log.debug("choose by throughput %s", suggest.concurrency)
        return suggest
    # select max distance(depth)
    diff = [(x, prevc[x + 1].concurrency - prevc[x].concurrency) for x in range(len(prevc) - 1)]
    diff.sort(key=lambda f: f[-1])
    idx = diff[-1][0]
    suggest = Cparam(int((prevc[idx].concurrency + prevc[idx + 1].concurrency) / 2))
    if select_chk(results, suggest):
        _log.debug("choose by depth %s", suggest.concurrency)
        return suggest
    # select max distance(latency)
    prevl = sorted(prevr, key=lambda f: f.latency)
    diff = [(x, prevl[x + 1].latency - prevl[x].latency) for x in range(len(prevl) - 1)]
    diff.sort(key=lambda f: f[-1])
    idx = diff[-1][0]
    suggest = Cparam(int((prevl[idx].concurrency + prevl[idx + 1].concurrency) / 2))
    if select_chk(results, suggest):
        _log.debug("choose by latency %s", suggest.concurrency)
        return suggest
    return None


def runcmd(cmd: str, args: list[str], stdout: Path, stderr: Path, timeout: Decimal | int | float, dry: bool) -> bool:
    if stdout.exists() and stdout.stat().st_size != 0:
        _log.info("skip running %s args=%s", cmd, args)
        return True
    _log.info("running %s, args=%s, stdout=%s, timeout=%s, dry=%s", cmd, args, stdout, timeout, dry)
    if not dry:
        try:
            with open(stdout, "w") as outfp, open(stderr, "w") as errfp:
                subprocess.check_call(
                    [cmd, *args], stdin=subprocess.DEVNULL, stdout=outfp, stderr=errfp, timeout=float(timeout)
                )
                return True
        except subprocess.CalledProcessError:
            _log.warning("command failed: stdout=%s, stderr=%s", stdout.read_text(), stderr.read_text())
            return False
        except FileNotFoundError:
            _log.error("command not found?: stdout=%s, stderr=%s", stdout.read_text(), stderr.read_text())
            raise
        except Exception:
            _log.error("something went wrong: stdout=%s, stderr=%s", stdout.read_text(), stderr.read_text())
            raise
    return False


@cli.command()
@verbose_option
@config_option
@click.option("--interval", type=float)
@click.option("--duration", type=TimeParam(), default=Decimal(30), show_default=True)
@click.option("--method", default="GET", show_default=True)
@click.option("--output", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option("--count", type=int, default=10, show_default=True)
@click.option("--min-depth", default=1, show_default=True)
@click.option("--max-depth", default=100, show_default=True)
@click.option("--dry/--wet", default=True, show_default=True)
@click.argument("url")
def run_simple1(
    config: dict,
    output: str,
    count: int,
    min_depth: int,
    max_depth: int,
    interval: float,
    duration: Decimal,
    method: str,
    url: str,
    dry: bool,
):
    """run benchmarks (simple1)"""
    cmd = config.get("command")
    if not cmd:
        raise click.Abort(f"invalid command: {config}")
    args = config.get("run", [])
    tmpl_args = dict(
        duration_sec=duration,
        duration_minutes=duration * 60,
        method=method,
        url=url,
        cpus=os.cpu_count(),
    )
    outdir = Path(output)
    for c in range(min_depth, max_depth, int((max_depth - min_depth) / count)):
        tmpl_args["concurrency"] = c
        cmdargs = [Template(x).render(tmpl_args) for x in args]
        outbase = outdir / f"{cmd}-{c}"
        success = runcmd(
            cmd,
            cmdargs,
            stdout=outbase.with_suffix(".stdout"),
            stderr=outbase.with_suffix(".stderr"),
            timeout=duration + 10,
            dry=dry,
        )
        if success and not dry:
            res1 = parse1(config, outbase.with_suffix(".stdout"))
            _log.info("results: %s", res1)
        if interval:
            _log.info("sleep %s", interval)
            if not dry:
                time.sleep(interval)


@cli.command()
@verbose_option
@config_option
@click.option("--interval", type=float)
@click.option("--duration", type=TimeParam(), default=Decimal(30), show_default=True)
@click.option("--method", default="GET", show_default=True)
@click.option("--output", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option("--count", type=int, default=10, show_default=True)
@click.option("--limit", type=float, default=0.3, show_default=True)
@click.option("--dry/--wet", default=True, show_default=True)
@click.option("--latency-key", default="99%tile", show_default=True)
@click.option("--throughput-key", default="reqs", show_default=True)
@click.argument("url")
def run_simple2(
    config: dict,
    output: str,
    limit: float,
    count: int,
    interval: float,
    duration: Decimal,
    method: str,
    url: str,
    dry: bool,
    latency_key: str,
    throughput_key: str,
):
    """run benchmarks (simple2)"""
    cmd = config.get("command")
    if not cmd:
        raise click.Abort(f"invalid command: {config}")
    args = config.get("run", [])
    tmpl_args = dict(
        duration_sec=duration,
        duration_minutes=duration / 60,
        method=method,
        url=url,
        cpus=os.cpu_count(),
    )
    cparam = []
    outdir = Path(output)
    for _ in range(count):
        nextp = select_cparam(cparam, Decimal(limit))
        if nextp is None:
            break
        tmpl_args["concurrency"] = nextp.concurrency
        cmdargs = [Template(x).render(tmpl_args) for x in args]
        outbase = outdir / f"{cmd}-{nextp.concurrency}"
        success = runcmd(
            cmd,
            cmdargs,
            stdout=outbase.with_suffix(".stdout"),
            stderr=outbase.with_suffix(".stderr"),
            timeout=duration + 10,
            dry=dry,
        )
        if success and not dry:
            res1 = parse1(config, outbase.with_suffix(".stdout"))
            nextp.throughput = Decimal(res1.get(throughput_key, Decimal(0)))
            nextp.latency = Decimal(res1.get(latency_key, Decimal(0)))
            _log.info("results: %s", nextp)
        cparam.append(nextp)
        if interval:
            _log.info("sleep %s", interval)
            if not dry:
                time.sleep(interval)


def runfunc(
    c,
    cmd: str,
    args: list[str],
    config: dict,
    interval: float,
    limit: float,
    tmpl_args: dict,
    outdir: Path,
    dry: bool,
    latency_key: str,
    throughput_key: str,
):
    targs = tmpl_args.copy()
    targs["concurrency"] = int(c[0])
    _log.info("targs %s", targs)
    cmdargs = [Template(x).render(targs) for x in args]
    outbase = outdir / f"{cmd}-{int(c[0])}"
    success = runcmd(
        cmd,
        cmdargs,
        stdout=outbase.with_suffix(".stdout"),
        stderr=outbase.with_suffix(".stderr"),
        timeout=targs["duration_sec"] + 10,
        dry=dry,
    )
    if interval:
        _log.info("sleep %s", interval)
        if not dry:
            time.sleep(interval)
    if success and not dry:
        res1 = parse1(config, outbase.with_suffix(".stdout"))
        _log.info("results: %s", res1)
        if res1[latency_key] < limit:
            _log.info("func return: %s", res1[throughput_key])
            return -float(res1[throughput_key])
    _log.info("func return: %s", 0.0)
    return 0.0


@cli.command()
@verbose_option
@config_option
@click.option("--interval", type=float)
@click.option("--duration", type=TimeParam(), default=Decimal(30), show_default=True)
@click.option("--method", default="GET", show_default=True)
@click.option("--output", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option("--count", type=int, default=10, show_default=True)
@click.option("--min-depth", default=1, show_default=True)
@click.option("--max-depth", default=100, show_default=True)
@click.option("--limit", type=float, default=0.3, show_default=True)
@click.option("--dry/--wet", default=True, show_default=True)
@click.option("--latency-key", default="99%tile", show_default=True)
@click.option("--throughput-key", default="reqs", show_default=True)
@click.option("--optimizer", type=click.Choice(["gp", "gbrt", "forest", "dummy"]), default="gp", show_default=True)
@click.argument("url")
def run_gp(
    config: dict,
    output: str,
    limit: float,
    count: int,
    min_depth: int,
    max_depth: int,
    interval: float,
    duration: Decimal,
    method: str,
    url: str,
    dry: bool,
    latency_key: str,
    throughput_key: str,
    optimizer: str,
):
    """run benchmarks (gp_minimize)"""
    cmd = config.get("command")
    if not cmd:
        raise click.Abort(f"invalid command: {config}")
    args = config.get("run", [])
    tmpl_args = dict(
        duration_sec=duration,
        duration_minutes=duration / 60,
        method=method,
        url=url,
        cpus=os.cpu_count(),
    )
    if optimizer == "gp":
        from skopt import gp_minimize as any_minimize
    elif optimizer == "gbrt":  # does not work
        from skopt import gbrt_minimize as any_minimize
    elif optimizer == "forest":
        from skopt import forest_minimize as any_minimize
    elif optimizer == "dummy":  # does not work
        from skopt import dummy_minimize as any_minimize

    outdir = Path(output)
    _func = functools.partial(
        runfunc,
        cmd=cmd,
        args=args,
        config=config,
        interval=interval,
        limit=limit,
        tmpl_args=tmpl_args,
        outdir=outdir,
        dry=dry,
        latency_key=latency_key,
        throughput_key=throughput_key,
    )
    result = any_minimize(_func, [(min_depth, max_depth)], n_calls=count)
    if result:
        click.echo(f"result: {-result.fun}")


def read_dir(inputd: Path, prefix: str, throughput_key: str, latency_key: str, config: dict):
    res = []
    files = sorted(inputd.glob(prefix + "-*.stdout"), key=lambda f: int(f.stem.split("-")[-1]))
    for file in files:
        data = parse1(config, file)
        if throughput_key in data and latency_key in data:
            ent = dict(
                concurrency=int(file.stem.split("-")[-1]),
                throughput=data.get(throughput_key),
                latency=data.get(latency_key),
            )
            res.append(ent)
    return res


@cli.command()
@verbose_option
@config_option
@format_option(default="csv")
@click.option("--latency-key", default="99%tile", show_default=True)
@click.option("--throughput-key", default="reqs", show_default=True)
@click.argument("input", type=click.Path(dir_okay=True, file_okay=False, exists=True))
def summary(config: dict, input: str, latency_key: str, throughput_key: str):
    """summarize throughput/latency"""
    inputd = Path(input)
    prefix = config.get("command", "")
    return read_dir(inputd=inputd, prefix=prefix, throughput_key=throughput_key, latency_key=latency_key, config=config)


@cli.command()
@verbose_option
@config_option
@format_option(default="csv")
@click.option("--latency-key", default="99%tile", show_default=True)
@click.option("--throughput-key", default="reqs", show_default=True)
@click.option("--limit", type=float, default=0.3, show_default=True)
@click.argument("input", type=click.Path(dir_okay=True, file_okay=False, exists=True), nargs=-1)
def result_all(config: dict, input: tuple[str], latency_key: str, throughput_key: str, limit: float):
    """summarize max throughput"""
    prefix = config.get("command", "")
    ret = []

    for i in input:
        inputd = Path(i)
        res = read_dir(
            inputd=inputd, prefix=prefix, throughput_key=throughput_key, latency_key=latency_key, config=config
        )
        if len(res) == 0:
            continue
        maxv = max([x for x in res if x["latency"] < limit], key=lambda f: f["throughput"])
        ret.append(dict(name=inputd.name, **maxv))
    return ret


@cli.command()
@verbose_option
@config_option
@click.option("--latency-key", default="99%tile", show_default=True)
@click.option("--throughput-key", default="reqs", show_default=True)
@click.option("--title")
@click.option("--limit", type=float, default=0.3, show_default=True)
@click.option("--save")
@click.argument("input", type=click.Path(dir_okay=True, file_okay=False, exists=True), nargs=-1)
def plot(config: dict, input: tuple[str], latency_key: str, throughput_key: str, title: str, limit: float, save: str):
    """plot throughput/latency"""
    import matplotlib.pyplot as plt

    prefix = config.get("command", "")
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylim(0, limit * 2)
    ax.axhline(limit, color="tab:green", ls="--")
    ax.set_xlabel("throughput (requests/s)")
    ax.set_ylabel("latency (second)")
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.grid()
    for i in input:
        inputd = Path(i)
        res = read_dir(
            inputd=inputd, prefix=prefix, throughput_key=throughput_key, latency_key=latency_key, config=config
        )
        if len(res) == 0:
            continue
        maxv = max([x for x in res if x["latency"] < limit], key=lambda f: f["throughput"])
        X = [x["throughput"] for x in res]
        Y = [x["latency"] for x in res]
        _log.debug("plot %s / %s", X, Y)
        ax.axvline(maxv["throughput"], ls=":")
        ax.plot(X, Y, "o-", label=inputd.name)
    ax.legend(loc=2)
    if save:
        plt.savefig(save)
    else:
        plt.show()


@cli.command()
@verbose_option
@config_option
@click.option("--title")
@click.option("--latency-key", default="99%tile", show_default=True)
@click.option("--throughput-key", default="reqs", show_default=True)
@click.option("--limit", type=float, default=0.3, show_default=True)
@click.option("--save")
@click.argument("input", type=click.Path(dir_okay=True, file_okay=False, exists=True), nargs=-1)
def plot_all(
    config: dict, input: tuple[str], latency_key: str, throughput_key: str, limit: float, title: str, save: str
):
    """plot max throughput"""
    import matplotlib.pyplot as plt

    prefix = config.get("command", "")
    ret = []

    for i in input:
        inputd = Path(i)
        res = read_dir(
            inputd=inputd, prefix=prefix, throughput_key=throughput_key, latency_key=latency_key, config=config
        )
        if len(res) == 0:
            continue
        maxv = max([x for x in res if x["latency"] < limit], key=lambda f: f["throughput"])
        ret.append(dict(name=inputd.name, **maxv))

    fig, ax = plt.subplots()
    ret.sort(key=lambda f: f["throughput"])
    ax.set_title(title)
    ax.set_xlabel("throughput (request/sec)")
    ax.grid(axis="x")
    X = [x["name"] for x in ret]
    Y = [x["throughput"] for x in ret]
    ax.barh(X, Y)
    if save:
        plt.savefig(save)
    else:
        plt.show()


@cli.command()
@verbose_option
@config_option
def install(config: dict):
    """show install command"""
    import distro

    search_path = [distro.id(), distro.like(), "binary", "source"]

    install_cmd = None
    for i in search_path:
        if install_cmd:
            break
        install_cmd = config.get("install", {}).get(i)

    if install_cmd:
        click.echo(install_cmd)
    else:
        click.echo(f"not found: {distro.id()} / {distro.like()}")


if __name__ == "__main__":
    cli()
