#!/usr/bin/env python3
"""
imgscaper — Generate unique 2D textures from example images and a specification.

Copyright (c) 2014, Alex J. Champandard
"""

import sys
import heapq
import random
import itertools

import numpy as np
import scipy.misc

import ops


PATCH_SIZE   = 128
PATCH_MIDDLE = PATCH_SIZE//2-1
PATCH_FINISH = PATCH_SIZE//2
PATCH_START  = -PATCH_FINISH+1

BIN_COUNT    = 128


class ImageScaper(object):
    """Main functionality used to generate images, either one static image if called
    from a script or a sequence of images if used as an API.
    """

    def __init__(self, source, spec):
        self.createMask()

        self.img = source
        self.img_size = source.shape
        self.img_blurred = ops.blur(source, 24.0)
        self.img_luminosity = self.img_blurred[:,:,1]

        L = self.img_luminosity
        hist, bins = np.histogram(L, density=True, bins=BIN_COUNT)
        L_indices = np.digitize(L.flatten(), bins)

        coordinates = np.indices((source.shape[0], source.shape[1])).swapaxes(0,2).swapaxes(0,1)
        self.c_coords = self._bin(bins, L_indices, coordinates)

        c_buckets = self._bin(bins, L_indices, self.img_blurred)
        c_averages = [np.average(bucket, axis=0) for bucket in c_buckets]

        ml = min(L.flatten())
        sl = max(L.flatten()) - ml

        self.spec = spec
        self.spec_normalized = (1.0 - spec ** 0.25) * sl + ml
        S_indices = np.digitize(self.spec_normalized.flatten(), bins)
        self.spec_bins = list(enumerate(S_indices))

        self.output = np.array([c_averages[(i-1)%BIN_COUNT] for i in S_indices], dtype=np.float32)\
                            .reshape(self.spec.shape[0], self.spec.shape[1], 3)


    def process(self, iterations):
        for i in range(iterations):
            sys.stdout.write("%3.1f%%\r" % (float(i+1) * 100.0 / iterations)); sys.stdout.flush();

            i, bn = random.choice(self.spec_bins)

            ty, tx = i//self.spec.shape[1], i%self.spec.shape[1]
            if ty+PATCH_START < 0 or ty+PATCH_FINISH > self.output.shape[0]:
                continue
            if tx+PATCH_START < 0 or tx+PATCH_FINISH > self.output.shape[1]:
                continue

            if bn-1 >= BIN_COUNT-1 or len(self.c_coords[bn-1]) == 0:
                continue

            sy, sx = self.pickBestPatch(ty, tx, self.c_coords[bn-1])
            if sx == -1 or sy == -1:
                continue
            self.splatThisPatch(sy, sx, ty, tx)

        repro = self.output.reshape(self.spec.shape[0], self.spec.shape[1], 3)
        return repro


    def _bin(self, bins, indices, array):
        flat_array = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
        return [flat_array[indices == i] for i in range(1, len(bins))]


    def createMask(self):
        mask_x = np.array([abs(x-PATCH_MIDDLE) for y, x in itertools.product(range(PATCH_SIZE-1), repeat=2)], dtype=np.float32) / (PATCH_FINISH-1)
        mask_y = np.array([abs(y-PATCH_MIDDLE) for y, x in itertools.product(range(PATCH_SIZE-1), repeat=2)], dtype=np.float32) / (PATCH_FINISH-1)

        mask_x = mask_x.reshape(PATCH_SIZE-1, PATCH_SIZE-1)
        mask_y = mask_y.reshape(PATCH_SIZE-1, PATCH_SIZE-1)

        mask = 2.0 * (1.0 - mask_x) * (1.0 - mask_y)
        mask[mask > 1.0] = 1.0

        self.mask = mask


    def D(self, sy, sx, ty, tx):
        return ((self.img[sy+PATCH_START:sy+PATCH_FINISH,sx+PATCH_START:sx+PATCH_FINISH]
               - self.output[ty+PATCH_START:ty+PATCH_FINISH,tx+PATCH_START:tx+PATCH_FINISH])**2).sum()


    def splatThisPatch(self, sy, sx, ty, tx):
        self.output[ty+PATCH_START:ty+PATCH_FINISH,tx+PATCH_START:tx+PATCH_FINISH,1] = \
            self.output[ty+PATCH_START:ty+PATCH_FINISH,tx+PATCH_START:tx+PATCH_FINISH,1] * (1.0 - self.mask) \
          + self.img[sy+PATCH_START:sy+PATCH_FINISH,sx+PATCH_START:sx+PATCH_FINISH,1] * self.mask


    def pickBestPatch(self, ty, tx, coords):
        results = []
        for sy, sx in random.sample(list(coords), min(len(coords), 100)):
            if sy+PATCH_START < 0 or sy+PATCH_FINISH > self.img.shape[0]:
                continue
            if sx+PATCH_START < 0 or sx+PATCH_FINISH > self.img.shape[1]:
                continue 
            d = self.D(sy, sx, ty, tx)
            heapq.heappush(results, (d, len(results), (sy,sx)))
        
        if not results:
            return -1, -1
        choices = heapq.nsmallest(5, results)
        return random.choice(choices)[2]



def main(args):
    if len(args) != 3:
        print("ERROR: Provide [spec] [source] [target] as script parameters.", file=sys.stderr)
        return -1

    spec = scipy.misc.imread(args[0])
    spec = ops.normalized(ops.distance(spec))

    src = ops.rgb2hls(scipy.misc.imread(args[1]))

    scraper = ImageScaper(src, spec)
    repro = scraper.process(5000)

    output = ops.hls2rgb(repro)
    scipy.misc.imsave(args[2], output)


if __name__ == "__main__":
    main(sys.argv[1:])
