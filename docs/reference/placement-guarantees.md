# Placement guarantees

What the placement UI promises, and how each promise is held and checked. This
is a reference for people changing the engine; the UI shows only the outcome.

| Promise | Held by | Checked by |
|---|---|---|
| A weight is where the plan says | The plan compiled to llama.cpp settings | Verifier, after every load (`crates/llama/src/verify.rs`) |
| The RAM tier stays in RAM | OS backstop, plus a driver feature when present | RAM tier status (`crates/worker/src/ram_tier.rs`) |
| The disk tier is a performance floor, not a hold | Nothing: it is the mapped file | Residency (`mincore`) |
| A model's host side is on its socket | CPU affinity, memory policy, strict `mbind` | NUMA report (`crates/worker/src/numa.rs`) |

Each check runs in the worker and reaches the UI in the worker's status.

## Weights are where they were planned

llama.cpp's own log misreports placement (decision D4), so it isn't read. After
loading, the worker evaluates a short batch in a separate context with a
scheduler callback. For every weight a node reads, it records the buffer and
address, and classifies it: inside the model file's mapping (disk tier), inside
one of our pinned allocations (RAM tier), in a GPU buffer (that GPU), or
elsewhere by buffer name. Run-time copies that the scheduler makes (op offload)
are attributed to the weight they copy. The result is compared with the plan;
`misplaced` counts the differences and the UI shows it. A mismatch is a bug.

## The RAM tier stays in RAM

Two layers. The OS-level one is the guarantee: no swap on the machine, the
buffer `mlock`ed, swap forbidden for the worker's cgroup (`memory.swap.max = 0`,
which the daemon arranges through a systemd user scope when it can), or
reserved huge pages (never swapped). Driver features (CUDA host registration)
are used for speed and also keep pages in RAM, but are listed separately and
don't count as the guarantee. When no OS-level hold works, the status says so
and gives the fix for how the engine is running (service, user session,
container, shell). Resident and swapped bytes are measured from `smaps`.

## The disk tier is a performance floor

Weights on the disk tier stay in the memory-mapped model file. The kernel may
drop them from RAM and read them again. Nothing holds them. What is promised is
only that they are never forced out: the status reports how many bytes are in
RAM right now (`mincore`), so a user can see what the tier costs.

## The host socket

On a machine with several sockets (NUMA nodes), a model's **host side** runs on
one socket: its RAM-tier memory, its CPU threads, and host-side allocations
(KV cache and compute buffers on the host, page cache pages it reads). The UI's
"RAM 0" and "CPU 0" boxes depend on this being true. llama.cpp doesn't
partition CPU work by node, so a model is never split across sockets.

### Choosing the node

`host_node` on the intent (carried into the plan; absent in older saved
intents, which still load) is the user's choice. When it is unset, the worker
picks:

1. the node of the GPU holding the most of the model's layers (lowest GPU index
   on a tie);
2. else, if no layer is on a GPU, GPU 0's node;
3. else (no GPUs), the socket with the most free memory.

A GPU's node comes from `/sys/bus/pci/devices/<pci id>/numa_node`, with the PCI
id from ggml's device properties (`device_id`). `-1` or a missing file means
node 0. Memory-only nodes (no CPUs, e.g. CXL) can't host a model; they are
never picked, and choosing one is refused. A node that doesn't exist is
refused, by the API when saving and by the worker when loading, with a message
like "socket 1 doesn't exist: this machine has one socket (socket 0)".

### Mechanism

The worker does it unprivileged, itself, before llama.cpp allocates the model
or starts threads:

- **CPU affinity**: `sched_setaffinity` on every existing thread to the node's
  CPUs, intersected with what the process may already use. New threads inherit
  it, and the runtime's thread-start hook reapplies it to threads the async
  runtime creates later. If none of the node's CPUs are allowed (e.g. started
  under `taskset` elsewhere) the load is refused.
- **Memory policy**: `set_mempolicy` with `MPOL_PREFERRED` for the node. It is
  tolerant: a full node spills to another instead of killing the process. It
  covers host-side buffers and page cache pages the model faults in.
- **RAM tier**: the pinned buffer is `mbind`ed strictly (`MPOL_BIND`) to the
  node before its first touch. This is only done on machines with several
  sockets, so single-socket machines run exactly as before.
- **Threads**: when the config doesn't set a thread count, one per CPU of the
  node.
- llama.cpp's own NUMA strategy (`llama_numa_init`) is never called, so it
  can't fight the affinity.

No cgroup cpuset is used: it isn't delegated to unprivileged users.

For automatic placement, llama.cpp's fit runs before the node is known (it needs
the plan), so the fit's dry run is unpinned; the real load is pinned.

### What is measured

Every status call (cached for two seconds) the worker reports:

- the affinity of every thread: how many CPUs, which sockets they belong to,
  and whether they are all on the chosen one;
- the memory policy that was set and whether it read back;
- the RAM tier's bytes per node, from `move_pages` (without a destination) on
  up to 8192 evenly spaced pages of each buffer (exact when the buffer has no
  more pages than that; an estimate otherwise);
- the disk tier's cached pages per node (the mapped model file, `numa_maps`
  `file=` lines), and the host buffers per node (anonymous memory that isn't
  the RAM tier: KV cache, compute buffers);
- all of the process's mapped memory per node, from `/proc/self/numa_maps`.

### What is claimed for the socket

The UI puts a model's RAM tier in the socket's RAM box and the help text says
host-side buffers and page cache pages it reads are on the socket too. So the
claim, and what is checked, is: the RAM tier, the host buffers, and the
model file's cached pages (the disk tier's in-RAM share). Other memory of the
process (libraries, other files) isn't claimed and isn't checked.

Memory policy only decides where *new* pages go. A page cache page has one
physical copy and stays on whichever node first read it, even for a different
process earlier. So after loading, and on every status measurement that finds
cached model-file pages on another node, the worker moves them: pages cached
but not yet mapped are mapped (`MADV_POPULATE_READ`, only the resident runs,
no I/O), then `mbind(MPOL_PREFERRED, MPOL_MF_MOVE)` over them; the mapping also
keeps the preference for pages read later. Then it measures again.

What can't be guaranteed: `MPOL_MF_MOVE` skips pages another process maps too
(moving those needs `CAP_SYS_NICE`). Two workers on different sockets using the
same model file can't both have it local; the one that doesn't hold the pages
reports it, naming the other process as the likely cause. Pages read after the
last measurement can also land elsewhere if another process cached them first;
the next status measurement moves them again.

### When it isn't where promised

On several sockets, the report turns into plain-language problems, shown in the
placement panel and added to the load warnings, for example "4.1 GB of the RAM
tier is on socket 1, not socket 0", "some threads of this model may run on
socket 1", "0.6 GB of the disk tier's cached pages are on socket 1, not socket
0" (with the reason and fix), or that more than a tenth of the host buffers
(and over 256 MiB) are on another socket. Memory that isn't claimed (libraries,
unrelated files) is never reported. Failures of the syscalls themselves are reported the same
way. A load isn't refused for them, except for the cases above. On one socket,
nothing is ever reported as a problem.

### Verification status

Verified on single-socket hardware only: the container used for development has
one node, where the worker pins to node 0's CPUs and reports it, a node that
doesn't exist is refused, and older saved intents load (end-to-end test). The
multi-socket logic (choosing a node, cpulist and PCI node parsing, problem
wording, memory per node from `numa_maps`) is covered by unit tests with fake
sysfs trees. The strict `mbind` path and the real placement of pages across two
nodes, and the PCI id lookup on real GPUs, have not been exercised on a
multi-socket machine.
