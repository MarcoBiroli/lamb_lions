"""Common style of the figures: colours, fonts, and where the figures are written (../figs, the folder read by the LaTeX file)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#898781', '#fcfcfb'        # text and lines, secondary text, axes, near-white
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'                      # lions, lamb, third series
SEQ = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
cmap = LinearSegmentedColormap.from_list('seq', SEQ)                       # one hue, light to dark: magnitude of the ground state

plt.rcParams.update({'font.size': 8, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.facecolor': 'white'})

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, '..', 'figs'))

def save(fig, name, **kw):
    """write figs/<name>.pdf"""
    os.makedirs(FIGDIR, exist_ok=True); path = os.path.join(FIGDIR, name + '.pdf'); fig.savefig(path, **kw); plt.close(fig)
    print("written:", path)
