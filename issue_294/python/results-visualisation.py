import os
import sys
import numpy
import math

from matplotlib import pyplot, cm

subsets = ['train', 'test']
counter_ids = ['TN', 'FP', 'FN', 'TP']


if __name__ == '__main__':
    #
    title = 'not specified'
    filenames = list()
    #
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--title':
            title = sys.argv[i + 1]
            i += 1
        else:
            filenames.append(sys.argv[i])
        i += 1
    #
    #
    counters = dict()
    for s in subsets:
        counters[s] = dict()
        for ci in counter_ids:
            counters[s][ci] = list()
    #
    '''
    states:
        0 -> nothing
        1 -> next line contains TN and FP
        2 -> next line contains FN and TP
    '''
    state = 0
    #
    for filename in filenames:
        f = open(filename, 'rt')
        subset = subsets[0]
        for line in f:
            if len(line) > 1:
                tokens = line.split()
                #Confusion matrix for subset train:
                #647 2
                #79 63
                if tokens[0] == 'Confusion' and tokens[1] == 'matrix':
                    state = 1
                elif state == 1:
                    counters[subset]['TN'].append(int(tokens[0]))
                    counters[subset]['FP'].append(int(tokens[1]))
                    state = 2
                elif state == 2:
                    counters[subset]['FN'].append(int(tokens[0]))
                    counters[subset]['TP'].append(int(tokens[1]))
                    subset = subsets[1] if subset == subsets[0] else subsets[0]
                    state = 0
                #0      645         4   TN, FP
                #1       15       127   FN, TP
                elif tokens[0] == '0' and tokens[3] == 'TN,' and tokens[4] == 'FP':
                    counters[subset]['TN'].append(int(tokens[1]))
                    counters[subset]['FP'].append(int(tokens[2]))
                elif tokens[0] == '1' and tokens[3] == 'FN,' and tokens[4] == 'TP':
                    counters[subset]['FN'].append(int(tokens[1]))
                    counters[subset]['TP'].append(int(tokens[2]))
                    subset = subsets[1] if subset == subsets[0] else subsets[0]
        f.close()
    #
    for subset in subsets:
        print(subset, ':')
        for ci in counter_ids:
            counters[subset][ci] = numpy.array(counters[subset][ci])
            x = counters[subset][ci]
            print('      ', ci, ':  %8.3f  %7.3f' % (x.mean(), x.std()))
        print()
    #
    #
    kargs = {'horizontalalignment': 'center',
             'verticalalignment': 'center',
             'fontsize': 10.0,
             'color': 'white'}
    #
    fig, axes = pyplot.subplots(nrows = 1, ncols = len(subsets))
    l = 50
    jet = cm.get_cmap('jet', 100)
    for s in range(len(subsets)):
        subset = subsets[s]
        d = counters[subset]
        axis = axes[s]
        axis.set_title(subset)
        z = numpy.zeros([2 * l, 2 * l, 3])
        for c in range(len(counter_ids)):
            ci = counter_ids[c]
            i = c // 2
            j = c % 2
            mean = d[ci].mean()
            sigma = d[ci].std()
            z[i * l : (i + 1) * l, j * l : (j + 1) * l, :] = jet(sigma / 4)[:3]
        z[:, l, :] = (1.0, 1.0, 1.0)
        z[l, :, :] = (1.0, 1.0, 1.0)
        im = axis.imshow(z)
        axis.set_yticks([])
        axis.set_xticks([])
        for c in range(len(counter_ids)):
            ci = counter_ids[c]
            i = c // 2
            j = c % 2
            mean = d[ci].mean()
            sigma = d[ci].std()
            y = i * l
            x = j * l
            kargs['color'] = 'black' if 0.3 < sigma / 4 < 0.7 else 'white'
                
            axis.annotate('$\mu = %.3f$'    % mean,  (x + 25, y + 20), **kargs)
            axis.annotate('$\sigma = %.3f$' % sigma, (x + 25, y + 30), **kargs)
            if i == 0 and j == 0:
                axis.annotate('TN', ( 5,  5), **kargs)
            elif i == 0 and j == 1:
                axis.annotate('FP', (95,  5), **kargs)
            elif i == 1 and j == 0:
                axis.annotate('FN', ( 5, 95), **kargs)
            elif i == 1 and j == 1:
                axis.annotate('TP', (95, 95), **kargs)
        #axis.grid()
    #cbar = fig.colorbar(im, extend = 'both', shrink = 1.0, ax = axis)
    pyplot.subplots_adjust(bottom = 0.0, top = 1.0, left = 0.05, right = 0.8)
    cax = pyplot.axes([0.85, 0.28, 0.020, 0.44])
    cbar = pyplot.colorbar(cm.ScalarMappable(cmap = 'jet'), cax = cax)
    cbar.set_label('standard deviation degree')
    cbar.set_ticks([])
    fig.suptitle(title)
    #pyplot.tight_layout()
    pyplot.show()
