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
BLUR_SIGMA   = 24.0
LIGHTNESS_POWER = 0.25

PATCH_COUNT = 100
BEST_COUNT = 5


class ImageScaper(object):
    """Main functionality used to generate images, either one static image if called
    from a script or a sequence of images if used as an API.
    """

    def __init__(self, source, spec):
        """Given a source image with interesting patterns and a specification that
        indicates desired luminosity (normalized), prepare an output image.

        Arguments:
            :source:    Image as 2D array of HLS-encoded pixels.
            :spec:      Image as 2D array of greyscale pixels.
        """

        # Source image is assumed to be a HLS-encoded array, which is blurred.
        self.img = source
        self.img_size = source.shape
        self.img_blurred = ops.blur(source, BLUR_SIGMA)
        self.img_luminosity = self.img_blurred[:,:,1]

        # Now we make a histogram of the blurred luminosities, each in bins.
        L = self.img_luminosity
        hist, bins = np.histogram(L, density=True, bins=BIN_COUNT)
        L_indices = np.digitize(L.flatten(), bins)

        # Store the center of all patches by using the luminosity bins. 
        coordinates = np.indices((source.shape[0], source.shape[1])).swapaxes(0,2).swapaxes(0,1)
        self.c_coords = self._bin(bins, L_indices, coordinates)

        # For each bin we calculate the average color, per-luminosity which assumes
        # the image patterns don't have too much hue variation.
        c_buckets = self._bin(bins, L_indices, self.img_blurred)
        c_averages = [np.average(bucket, axis=0) for bucket in c_buckets]

        # Normalize the specification image based on what our luminosity can provide.
        ml = min(L.flatten())
        sl = max(L.flatten()) - ml
        self.spec_normalized = (1.0 - spec ** LIGHTNESS_POWER) * sl + ml
        self.spec = spec

        # Apply the same binning process to the spec image....
        S_indices = np.digitize(self.spec_normalized.flatten(), bins)
        self.spec_bins = list(enumerate(S_indices))

        # Generate a first version of the output based on the average given the luminosity
        # of the specification.  There are no interesting patterns, just colors.
        self.output = np.array([c_averages[(i-1)%BIN_COUNT] for i in S_indices], dtype=np.float32)\
                            .reshape(self.spec.shape[0], self.spec.shape[1], 3)

        # Prepare a masking array used for blending and feathering out the edges of patches.
        self.createMask()


    def process(self, iterations):
        """Randomly pick locations to add patches to the output image, and pick the best 
        parts of the source image accordingly.
        """

        for i in range(iterations):
            sys.stdout.write("%3.1f%%\r" % (float(i+1) * 100.0 / iterations)); sys.stdout.flush();

            # Select a random pixel index (i) and determine its bin (bn).
            i, bn = random.choice(self.spec_bins)

            # Check coordinates and discard if it's out of bounds.
            # TODO: Apply the patch anyway with clamping/mirroring; check if numpy
            # supports this form of indexing.
            ty, tx = i//self.spec.shape[1], i%self.spec.shape[1]
            if ty+PATCH_START < 0 or ty+PATCH_FINISH > self.output.shape[0]:
                continue
            if tx+PATCH_START < 0 or tx+PATCH_FINISH > self.output.shape[1]:
                continue

            # In some cases the bins chosen may not contain any samples, in that case
            # just ignore this pixel and try again.
            if bn-1 >= BIN_COUNT-1 or len(self.c_coords[bn-1]) == 0:
                continue

            # Find a source image patch for this target coordinate, and then splat it!
            sy, sx = self.pickBestPatch(ty, tx, self.c_coords[bn-1])
            if sx == -1 or sy == -1:
                continue
            self.splatThisPatch(sy, sx, ty, tx)

        # The output image as HLS can now be used in its current form, or other
        # iterations may be performed.
        repro = self.output.reshape(self.spec.shape[0], self.spec.shape[1], 3)
        return repro


    def _bin(self, bins, indices, array):
        """Given a histogram's bin and a set of binned indices, select the subset of the
        array that correspons to each of the bins.
        """
        flat_array = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
        return [flat_array[indices == i] for i in range(1, len(bins))]


    def createMask(self):
        """Create a square mask for blending that fades out towards the edges and has a solid
        block of unblended pixels in the middle.
        """
        mask_x = np.array([abs(x-PATCH_MIDDLE) for y, x in itertools.product(range(PATCH_SIZE-1), repeat=2)], dtype=np.float32) / (PATCH_FINISH-1)
        mask_y = np.array([abs(y-PATCH_MIDDLE) for y, x in itertools.product(range(PATCH_SIZE-1), repeat=2)], dtype=np.float32) / (PATCH_FINISH-1)

        mask_x = mask_x.reshape(PATCH_SIZE-1, PATCH_SIZE-1)
        mask_y = mask_y.reshape(PATCH_SIZE-1, PATCH_SIZE-1)

        mask = 2.0 * (1.0 - mask_x) * (1.0 - mask_y)
        mask[mask > 1.0] = 1.0

        self.mask = mask


    def D(self, sy, sx, ty, tx):
        """Calculate the cost of blending this patch from the source image into the output
        image, based on the squared distance of each pixel component (HLS).
        """
        return ((self.img[sy+PATCH_START:sy+PATCH_FINISH,sx+PATCH_START:sx+PATCH_FINISH]
               - self.output[ty+PATCH_START:ty+PATCH_FINISH,tx+PATCH_START:tx+PATCH_FINISH])**2).sum()


    def splatThisPatch(self, sy, sx, ty, tx):
        """Store a patch centered on (ty, tx) in the output image based on the source
        image at location (sy, sx), using the blend mask calculated statically.
        """
        self.output[ty+PATCH_START:ty+PATCH_FINISH,tx+PATCH_START:tx+PATCH_FINISH,1] = \
            self.output[ty+PATCH_START:ty+PATCH_FINISH,tx+PATCH_START:tx+PATCH_FINISH,1] * (1.0 - self.mask) \
          + self.img[sy+PATCH_START:sy+PATCH_FINISH,sx+PATCH_START:sx+PATCH_FINISH,1] * self.mask


    def pickBestPatch(self, ty, tx, coords):
        """Iterate over a random selection of patches (e.g. 100) and pick a random
        sample of the best (e.g. top 5).  Distance metric is used to rank the patches.
        """
        results = []
        for sy, sx in random.sample(list(coords), min(len(coords), PATCH_COUNT)):
            # TODO: If the patch doesn't fully fit, then assume the image either
            # clamps or loops, depending what numpy can do easily.
            if sy+PATCH_START < 0 or sy+PATCH_FINISH > self.img.shape[0]:
                continue
            if sx+PATCH_START < 0 or sx+PATCH_FINISH > self.img.shape[1]:
                continue 
            d = self.D(sy, sx, ty, tx)
            heapq.heappush(results, (d, len(results), (sy,sx)))
        
        # Some unlucky cases with special images cause no patches to be found
        # at all, in this case we just bail out.
        if not results:
            return -1, -1

        choices = heapq.nsmallest(BEST_COUNT, results)
        return random.choice(choices)[2]



def main(args):
    if len(args) != 3:
        print("ERROR: Provide [spec] [source] [target] as script parameters.", file=sys.stderr)
        return -1

    spec = scipy.misc.imread(args[0])
    spec = ops.normalized(ops.distance(spec))
    scipy.misc.imsave('output_dist.jpg', spec)

    src = ops.rgb2hls(scipy.misc.imread(args[1]))

    scraper = ImageScaper(src, spec)
    repro = scraper.process(5000)

    output = ops.hls2rgb(repro)
    scipy.misc.imsave(args[2], output)


if __name__ == "__main__":
    main(sys.argv[1:])
