"""Enumerate ordinary ASCII HTTP(S) URLs encoded in one QR byte segment.

Usage: python3 solve_qr_urls.py
Requires numpy, qrcode, and reedsolo.
"""
from itertools import product
import json
from pathlib import Path
import re
import sys

import numpy as np
import qrcode
from qrcode import base, constants, util
from reedsolo import RSCodec


N = 29
MATRIX_PATH = Path('qr-matrix.json')
OUTPUT_QR_PATH = Path('recovered-qr.png')


def structure(level,mask):
    qr=qrcode.QRCode(version=3,error_correction=level)
    qr.modules_count=N
    qr.modules=[[None]*N for _ in range(N)]
    qr.setup_position_probe_pattern(0,0)
    qr.setup_position_probe_pattern(N-7,0)
    qr.setup_position_probe_pattern(0,N-7)
    qr.setup_position_adjust_pattern()
    qr.setup_timing_pattern()
    qr.setup_type_info(False,mask)
    fixed=np.array(qr.modules,dtype=object)
    # Encode distinct bits one at a time to recover the library's exact placement order.
    nbytes=sum(block.total_count for block in base.rs_blocks(3,level))
    zeros=[0]*nbytes
    qr.map_data(zeros,mask)
    baseline=np.array(qr.modules,dtype=bool)
    coords=[]
    for i in range(nbytes*8):
        qr.modules=fixed.tolist()
        data=zeros.copy()
        data[i//8]=128>>(i%8)
        qr.map_data(data,mask)
        coords.append(tuple(np.argwhere(baseline!=qr.modules)[0]))
    return fixed,baseline,np.array(coords).T


def blocks_for(level):
    blocks=base.rs_blocks(3,level)
    indices=[[] for _ in blocks]
    cursor=0
    for parity in (False,True):
        lengths=[b.total_count-b.data_count if parity else b.data_count for b in blocks]
        for i in range(max(lengths)):
            for b,length in enumerate(lengths):
                if i<length:
                    indices[b].append(cursor)
                    cursor+=1
    return [(np.array(ids), b.data_count, b.total_count-b.data_count) for ids,b in zip(indices,blocks)]


def equations(k,r):
    rs=RSCodec(r)
    rows=[0]*((k+r)*8)
    for i in range(k*8):
        data=bytearray(k)
        data[i//8]=128>>(i%8)
        for j,b in enumerate(rs.encode(data)):
            for bit in range(8):
                if b & (128>>bit):
                    rows[j*8+bit]|=1<<i
    return rows


def solve(rows, known, bits, return_basis=False):
    basis={}
    for i in np.flatnonzero(known):
        row,b=rows[i],int(bits[i])
        while row:
            pivot=row.bit_length()-1
            if pivot not in basis:
                basis[pivot]=(row,b)
                break
            coeff,rhs=basis[pivot]
            row^=coeff
            b^=rhs
        else:
            if b:
                return len(basis),None
    if return_basis:
        return len(basis), basis
    x=0
    for pivot,(row,b) in sorted(basis.items()):
        if ((row & x).bit_count() & 1)^b:
            x|=1<<pivot
    return len(basis), x


def complete_assignment(basis, free_bits):
    assignment = free_bits
    for pivot, (row, rhs) in sorted(basis.items()):
        if ((row & assignment).bit_count() & 1) ^ rhs:
            assignment |= 1 << pivot
    return assignment


URL = re.compile(
    rb"https?://(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
    rb"(?:[/?#][A-Za-z0-9._~:/?#@!$&'()*+,;=%-]*)?"
)


def byte_segment_constraints(prefix, length, capacity):
    buffer = util.BitBuffer()
    buffer.put(util.MODE_8BIT_BYTE, 4)
    buffer.put(length, 8)
    for value in prefix:
        buffer.put(value, 8)
    start = len(buffer)
    end = 12 + length * 8
    # Unknown content is assigned later. Its high bits are fixed to zero for ASCII.
    while len(buffer) < min(capacity, ((end + 11) // 8) * 8):
        buffer.put_bit(False)
    for i in range((capacity - len(buffer)) // 8):
        buffer.put((0xEC, 0x11)[i % 2], 8)
    fixed = np.ones(capacity, dtype=bool)
    fixed[start:end] = False
    fixed[start:end:8] = True
    return fixed, np.unpackbits(np.array(buffer.buffer, dtype=np.uint8))


def block_options(blocks, equations_by_block, known, observed, fixed, bits):
    options = []
    offset = 0
    for (ids, data_bytes, _), matrix in zip(blocks, equations_by_block):
        indices = (ids[:, None] * 8 + np.arange(8)).ravel()
        size = data_bytes * 8
        local = slice(offset, offset + size)
        _, basis = solve(
            matrix + matrix[:size],
            np.r_[known[indices], fixed[local]],
            np.r_[observed[indices], bits[local]],
            return_basis=True,
        )
        if basis is None:
            return None
        options.append((size, basis))
        offset += size
    return options


def data_assignments(size, basis):
    free = [i for i in range(size) if i not in basis]
    for selection in range(1 << len(free)):
        free_bits = sum(((selection >> j) & 1) << i for j, i in enumerate(free))
        value = complete_assignment(basis, free_bits)
        yield np.packbits([(value >> i) & 1 for i in range(size)]).tobytes()


def enumerate_urls(evidence):
    if evidence['schema'] != 'qr-module-evidence/v1' or evidence['version'] != 3:
        raise ValueError('Expected a Version 3 qr-module-evidence/v1 matrix')
    level = getattr(constants, 'ERROR_CORRECT_' + evidence['error_correction'])
    _, baseline, coords = structure(level, evidence['mask'])
    blocks = blocks_for(level)
    parity = [equations(k, r) for _, k, r in blocks]
    capacity = sum(k * 8 for _, k, _ in blocks)
    modules = np.array([list(row) for row in evidence['modules']])[tuple(coords)]
    known = np.isin(modules, ['0', '1'])
    observed = (modules == '1') ^ baseline[tuple(coords)]
    results = [set(), set()]
    stats = [[0, 0, 0], [0, 0, 0]]
    maximum_length = (capacity - 12) // 8
    print(f'矩阵：Version {evidence["version"]}，纠错 {evidence["error_correction"]}，'
          f'掩模 {evidence["mask"]}。已知模块 {known.sum()} 个，未知模块 {(~known).sum()} 个。', file=sys.stderr)
    for prefix in (b'http://', b'https://'):
        print(f'求解 {prefix.decode()}，尝试总长 {len(prefix)}–{maximum_length} 字节。', file=sys.stderr)
        for length in range(len(prefix), maximum_length + 1):
            fixed, bits = byte_segment_constraints(prefix, length, capacity)
            for wrong in [None, *np.flatnonzero(known)]:
                errors = int(wrong is not None)
                stats[errors][0] += 1
                corrected = observed.copy()
                if wrong is not None:
                    corrected[wrong] ^= True
                options = block_options(blocks, parity, known, corrected, fixed, bits)
                if options is None:
                    continue
                stats[errors][1] += 1
                for chunks in product(*(data_assignments(*option) for option in options)):
                    stats[errors][2] += 1
                    word = int.from_bytes(b''.join(chunks), 'big')
                    word >>= capacity - 12 - length * 8
                    value = (word & ((1 << (length * 8)) - 1)).to_bytes(length, 'big')
                    if URL.fullmatch(value):
                        location = None if wrong is None else tuple(int(i) + 1 for i in coords[:, wrong])
                        results[wrong is not None].add((value.decode('ascii'), location))
    for errors, (tried, solvable, payloads) in enumerate(stats):
        print(f'恰好 {errors} 个模块误判：尝试 {tried} 组，{solvable} 组有解，'
              f'得到 {payloads} 条字节串，{len(results[errors])} 条通过网址检查。', file=sys.stderr)
    return results


if __name__ == '__main__':
    evidence = json.loads(MATRIX_PATH.read_text(encoding='utf-8'))
    results = enumerate_urls(evidence)
    print(json.dumps({
        f'exactly_{n}_errors': [{'url': url, 'mismatch_row_column': location}
                               for url, location in sorted(group)]
        for n, group in enumerate(results)
    }, indent=2))

    urls = {url for group in results for url, _ in group}
    if len(urls) != 1:
        sys.exit('PNG export requires exactly one URL.')
    level = getattr(constants, 'ERROR_CORRECT_' + evidence['error_correction'])
    qr = qrcode.QRCode(
        version=evidence['version'], error_correction=level,
        mask_pattern=evidence['mask'], box_size=12, border=4,
    )
    qr.add_data(urls.pop(), optimize=0)
    qr.make(fit=False)
    qr.make_image().save(OUTPUT_QR_PATH)
    print(f'已输出二维码：{OUTPUT_QR_PATH}', file=sys.stderr)
