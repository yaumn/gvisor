# Debugging

[TOC]

To enable debug and system call logging, add the `runtimeArgs` below to your
[Docker](../quick_start/docker/) configuration (`/etc/docker/daemon.json`):

```json
{
    "runtimes": {
        "runsc": {
            "path": "/usr/local/bin/runsc",
            "runtimeArgs": [
                "--debug-log=/tmp/runsc/",
                "--debug",
                "--strace"
            ]
       }
    }
}
```

> Note: the last `/` in `--debug-log` is needed to interpret it as a directory.
> Then each `runsc` command executed will create a separate log file. Otherwise,
> log messages from all commands will be appended to the same file.

You may also want to pass `--log-packets` to troubleshoot network problems. Then
restart the Docker daemon:

```bash
sudo systemctl restart docker
```

Run your container again, and inspect the files under `/tmp/runsc`. The log file
ending with `.boot` will contain the strace logs from your application, which
can be useful for identifying missing or broken system calls in gVisor. If you
are having problems starting the container, the log file ending with `.create`
may have the reason for the failure.

## Stack traces

The command `runsc debug --stacks` collects stack traces while the sandbox is
running which can be useful to troubleshoot issues or just to learn more about
gVisor. It connects to the sandbox process, collects a stack dump, and writes it
to the console. For example:

```bash
docker run --runtime=runsc --rm -d alpine sh -c "while true; do echo running; sleep 1; done"
63254c6ab3a6989623fa1fb53616951eed31ac605a2637bb9ddba5d8d404b35b

sudo runsc --root /var/run/docker/runtime-runsc/moby debug --stacks 63254c6ab3a6989623fa1fb53616951eed31ac605a2637bb9ddba5d8d404b35b
```

> Note: `--root` variable is provided by docker and is normally set to
> `/var/run/docker/runtime-[runtime-name]/moby`. If in doubt, `--root` is logged
> to `runsc` logs.

## Debugger

You can debug gVisor like any other Golang program. If you're running with
Docker, you'll need to find the sandbox PID and attach the debugger as root.
Here is an example:

Install a runsc with debug symbols (you can also use the
[nightly release](../install/#nightly)):

```bash
make dev BAZEL_OPTIONS="-c dbg --define gotags=debug"
```

Start the container you want to debug using the runsc runtime with debug
options:

```bash
docker run --runtime=$(git branch --show-current)-d --rm --name=test -p 8080:80 -d nginx
```

Find the PID and attach your favorite debugger:

```bash
sudo dlv attach $(docker inspect test | grep Pid | head -n 1 | grep -oe "[0-9]*")
```

Set a breakpoint for accept:

```bash
break gvisor.dev/gvisor/pkg/sentry/socket/netstack.(*SocketOperations).Accept
continue
```

In a different window connect to nginx to trigger the breakpoint:

```bash
curl http://localhost:8080/
```

It's also easy to attach a debugger to one of the predefined syscall tests when
you're working on specific gVisor features. With the `delay-for-debugger` flag
you can pause the test runner before execution so that you can attach the
sandbox process to a debugger. Here is an example:

```bash
 make test BAZEL_OPTIONS="-c dbg --define gotags=debug" \
 OPTIONS="--test_arg=--delay-for-debugger=5m --test_output=streamed" \
 TARGETS=//test/syscalls:mount_test_runsc_systrap
```

The `delay-for-debugger=5m` flag means the test runner will pause for 5 minutes
before running the test. To attach to the sandbox process, you can run the
following in a separate window.

```bash
dlv attach $(ps aux | grep -m 1 -e 'runsc-sandbox' | awk '{print $2}')
```

Once you've attached to the process and set a breakpoint, you can signal the
test to start by running the following in another separate window.

```bash
kill -SIGUSR1 $(ps aux | grep -m 1 -e 'bash.*test/syscalls' | awk '{print $2}')
```

## Profiling

`runsc` integrates with Go profiling tools and gives you easy commands to
profile CPU and heap usage. First you need to enable `--profile` in the command
line options before starting the container:

```json
{
    "runtimes": {
        "runsc-prof": {
            "path": "/usr/local/bin/runsc",
            "runtimeArgs": [
                "--profile"
            ]
       }
    }
}
```

> Note: Enabling profiling loosens the seccomp protection added to the sandbox,
> and should not be run in production under normal circumstances.

Then restart docker to refresh the runtime options. While the container is
running, execute `runsc debug` to collect profile information and save to a
file. Here are the options available:

*   **--profile-heap:** Generates heap profile to the speficied file.
*   **--profile-cpu:** Enables CPU profiler, waits for `--duration` seconds and
    generates CPU profile to the speficied file.

For example:

```bash
docker run --runtime=runsc-prof --rm -d alpine sh -c "while true; do echo running; sleep 1; done"
63254c6ab3a6989623fa1fb53616951eed31ac605a2637bb9ddba5d8d404b35b

sudo runsc --root /var/run/docker/runtime-runsc-prof/moby debug --profile-heap=/tmp/heap.prof 63254c6ab3a6989623fa1fb53616951eed31ac605a2637bb9ddba5d8d404b35b
sudo runsc --root /var/run/docker/runtime-runsc-prof/moby debug --profile-cpu=/tmp/cpu.prof --duration=30s 63254c6ab3a6989623fa1fb53616951eed31ac605a2637bb9ddba5d8d404b35b
```

The resulting files can be opened using `go tool pprof` or [pprof][]. The
examples below create image file (`.svg`) with the heap profile and writes the
top functions using CPU to the console:

```bash
go tool pprof -svg /usr/local/bin/runsc /tmp/heap.prof
go tool pprof -top /usr/local/bin/runsc /tmp/cpu.prof
```

[pprof]: https://github.com/google/pprof/blob/master/doc/README.md

### Docker Proxy

When forwarding a port to the container, Docker will likely route traffic
through the [docker-proxy][]. This proxy may make profiling noisy, so it can be
helpful to bypass it. Do so by sending traffic directly to the container IP and
port. e.g., if the `docker0` IP is `192.168.9.1`, the container IP is likely a
subsequent IP, such as `192.168.9.2`.

[docker-proxy]: https://windsock.io/the-docker-proxy/
