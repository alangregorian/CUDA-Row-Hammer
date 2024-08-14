import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

L1CACHE_SIZE = 128
L2CACHE_SIZE = 96*1024
GPU_NAME = 'Nvidia L40S'

def load_data(filename: str) -> dict:
    try:
        data = {}
        with open(filename, 'r') as file:
            for line in file:
                if line.find(':') == -1:
                    size, latency = line.strip().split(',')
                    data[int(size)] = float(latency)

            file.close()

        return data
    except Exception as error:
        print(error)
        sys.exit(1)

def plot_latency(df: pd.DataFrame, filename: str):
    fig = plt.figure()
    fig.set_figwidth(10)

    ax = plt.gca()
    ax.set_xscale('log', base=2)
    ax.set_xticks(2 ** np.arange(21))
    ax.set_title('Cache and Memory Latency')
    ax.set_xlabel('Pointer Chasing Region (KB)')
    ax.set_ylabel('Latency (ns)')
    ax.set_ylim(0, 300)
    ax.margins(x=0)
    ax.grid(True)

    ax.annotate(f'{L1CACHE_SIZE}KB', (L1CACHE_SIZE*1.1, 70), fontsize=12)
    ax.annotate(f'{int(L2CACHE_SIZE/1024)}MB', (L2CACHE_SIZE*1.1, 70), fontsize=12)
    plt.vlines(x=[L1CACHE_SIZE, L2CACHE_SIZE], ymin=0, ymax=300, colors='black',
               linestyle='dashed', linewidth=1.5, label='_nolegend_')

    plt.plot('size', 'latency', data=df, color='green', linewidth=1.5)
    plt.legend(loc='upper left', labels=[GPU_NAME])

    plt.savefig(filename + '.png')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: python3 {sys.argv[0]} <filename>')
        sys.exit(1)

    filename = sys.argv[1]
    data = load_data(filename)
    df = pd.DataFrame(data.items(), columns = ['size', 'latency'])

    plot_latency(df, filename)
