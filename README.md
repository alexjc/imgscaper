ImageScaper
===========

A command-line tool for generating unique 2D textures procedurally given example images and a specification.  It's a patch-based algorithm that repeatedly splats areas from the example image into the generated output image.



![Input/Output Example](/docs/example.jpg?raw=true)


Usage
-----

### 1. Find Input Examples

The best types of inputs are high-resolution photos or textures that are significantly larger than the target output.  It's recommended you save these in a lossless format such as PNG.  Manually touching up or post-processing the example input also helps generate great looking output.

Currently, only images with relatively uniform hues (e.g. desert, ocean, rocks, forest, grass, etc.) will work well.  Image sizes up-to 2560x1440 were tested with the current code.

### 2. Specify Desired Output

The output image will match the resolution and luminosity of your input specification as a greyscale image.  Currently, all pixels must be specified (no alpha) and the source material will be drawn from the same image type.

### 3. Generate ImageScapes

Once you have all the dependencies installed, run the tool via Python or directly from a wrapper script:

    imgscaper input.jpg spec.jpg output.png

This takes about one minute to process currently, for an image of size 1920x1080 with a significantly larger input and the default parameters.


Description
-----------

Here's how the algorithm currently works:

1. The example input image is blurred using a gaussian filter and each of the resulting pixels are binned based on their luminosity.

2. The input example is broken up into patches stored in buckets based on the luminosity of the center pixel (blurred).

3. A first pass of the output is generated based on the input specification, looking up the average color (incl. hue and saturation) for the desired luminosity.

4. Random pixel coordinates are chosen in the target image, and a suitable patch is searched randomly from the source image; one of the top 5 best patches is selected.

5. The luminosity of the target image is replaced from the source image, but the hue and saturation are preserved as they were initially set.

6. After a specified number of iterations, the algorithm terminates and returns the image as it is.  Use the API to get animated images!

If you have any questions, feel free to ask me on Twitter as [@alexjc](http://twitter.com/alexjc).