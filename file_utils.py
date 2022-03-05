"""Utils for files handling.

Author: Lang Liu
"""

from __future__ import absolute_import, division, print_function

import numpy as np


def file_name_test(args):
    if args.experiment in ['bilingual', 'image-text']:
        fname = f'{args.experiment}-{args.data}-size{args.size}-'
    else:
        fname = f'{args.dist}-size{args.size}-dim{args.dim}-'

    if args.test == 'etic':
        fname += f'{args.test}-power{args.power}-reg{args.reg}m-nperms{args.nperms}'
    elif args.test == 'aetic':
        fname += f'{args.test}-power{args.power}-nperms{args.nperms}'
    elif args.test == 'baetic':
        fname += f'{args.test}-power{args.power}-nregs{args.nregs}-nperms{args.nperms}'
    elif args.test == 'rfetic':
        fname += f'{args.test}-power{args.power}-reg{args.reg}m-nperms{args.nperms}-nfeat{args.nfeat}'
        if args.npc:
            fname += f'-pca{args.npc}'
            if args.whiten:
                fname += '-whiten'
    elif args.test in ['hsic', 'hsic-adaptive']:
        fname += f'{args.test}-{args.kernel}-kpar{args.kpar}m-nperms{args.nperms}'
    else:
        raise ValueError('Invalid test name.')
    return fname


def concat_txt(fpath, tpath, prefix, parts):
    """Concatenate results from different txt files.

    Parameters
    ----------
    fpath: str
        Path of files to be concatenated.
    tpath: str
        Path of files to be stores.
    prefix : str
        Prefix of the file name. All files should be named as
        ``f"{prefix}-part-{i}.txt"`` for ``i in parts``.
    parts : array-like
        Numbering of all files.
    """

    num = 0
    for i, p in enumerate(parts):
        fname = f"{fpath}/{prefix}-part-{p}.txt"
        try:
            res = np.loadtxt(fname, delimiter=',')
            num += 1
        except OSError:
            continue
        if i == 0:
            Res = np.array(res)
        else:
            Res = np.concatenate((Res, res))
    fname = f"{tpath}/{prefix}.txt"
    np.savetxt(fname, Res)
    print(f"Results saved ({num}).")


def vstack_txt(prefix, parts, save=True):
    """Stack results by row from different txt files.

    Parameters
    ----------
    prefix : str
        Prefix of the file name. All files should be named as
        ``f"{prefix}-part-{i}.txt"`` for ``i in parts``.
    parts : array-like
        Numbering of all files.
    """

    for i, p in enumerate(parts):
        fname = f"{prefix}-part{p}.txt"
        res = np.loadtxt(fname, delimiter=',')
        if i == 0:
            Res = np.array(res)
        else:
            Res = np.vstack((Res, res))
    if save:
        fname = f"{prefix}.txt"
        np.savetxt(fname, Res, delimiter=',')
        print("Results saved.")
    return Res


def average_res(prefix, parts):
    res = []
    count = 0
    for part in parts:
        fname = f'tmp/{prefix}-part{part}.txt'
        try:
            res.append(np.loadtxt(fname))
            count += 1
        except OSError:
            continue
    print(f'{count} valid experimental results.')
    return np.mean(res, axis=0)
