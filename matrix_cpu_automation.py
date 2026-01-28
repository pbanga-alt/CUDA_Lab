import subprocess
import re
import matplotlib.pyplot as plt

times = []
Ns = []
pattern = re.compile(r":\s*([0-9.]+)\s*seconds")

for N in range(512,3584,512):
    result = subprocess.run(
        ["./matrix_cpu", str(N)],
        capture_output=True,
        text=True
    )

    match = pattern.search(result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse output for N={N}")

    t = float(match.group(1))
    times.append(t)
    Ns.append(N)
    print(f"N={N}, time={t} s")

plt.plot(Ns, times, marker="o")
plt.xlabel("N")
plt.ylabel("Runtime (seconds)")
plt.title("Matrix CPU Runtime")
plt.grid(True)
plt.show()
