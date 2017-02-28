from PIL import Image
import numpy

im = Image.open('/home/paul/Pictures/Ian-Wright-2.jpg', 'r')

#print image


#imageArray = numpy.fromstring(image.tostring(), dtype='uint8', count=-1, sep='').reshape(image.shape + (len(image.getbands()),))

s = im.tostring()
print len(s)
array1 = numpy.asarray(im)
print array1