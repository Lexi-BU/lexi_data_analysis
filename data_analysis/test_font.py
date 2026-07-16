import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["text.latex.preamble"] = r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"

plt.plot([1, 2], [3, 4])
plt.title("Test Arial-like Font")
plt.savefig("test_helvet.png")
