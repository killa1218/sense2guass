from scipy.stats import norm
from scipy.stats import entropy as kl
import matplotlib.pyplot as plt
import numpy as np

#initialize a normal distribution with frozen in mean=-1, std. dev.= 1
blue = norm(loc = -1., scale = 1.0)
red = norm(loc = -1., scale = 3.0)
black = norm(loc = 2., scale = 2.0)
cyan0 = norm(loc = 5., scale = 1.0)
cyan = norm(loc = 5., scale = 5.0)

green = norm(loc = 0., scale = 2.0)
purple = norm(loc = 0., scale = 4.0)
yellow = norm(loc = 0., scale = 5.0)

x = np.arange(-20, 20, .1)
y = np.arange(-20, 20, .1)

#plot the pdfs of these normal distributions
plt.plot(
    x, blue.pdf(x), 'b',
    x, green.pdf(x), 'g',
    x, red.pdf(x), 'r',
    x, cyan.pdf(x), 'c',
    x, purple.pdf(x), 'm',
    x, black.pdf(x), 'k',
    x, cyan0.pdf(x), 'c',
    x, yellow.pdf(x), 'y'
)


# print 'Blue to Green:', kl(blue.pdf(x), green.pdf(x))
# print 'Green to Blue:', kl(green.pdf(x), blue.pdf(x))
# print 'Green to Red:', kl(green.pdf(x), red.pdf(x))
# print 'Red to Green:', kl(red.pdf(x), green.pdf(x))
# print 'Red to Blue:', kl(red.pdf(x), blue.pdf(x))
# print 'Blue to Red:', kl(blue.pdf(x), red.pdf(x))

print('Green to Purple:', kl(green.pdf(x), purple.pdf(x)))
print('Purple to Green:', kl(purple.pdf(x), green.pdf(x)))
print('Green to Yellow:', kl(green.pdf(x), yellow.pdf(x)))
print('Yellow to Green:', kl(yellow.pdf(x), green.pdf(x)))
print('Cyan to Yellow:', kl(cyan.pdf(x), yellow.pdf(x)))
print('Yellow to Cyan:', kl(yellow.pdf(x), cyan.pdf(x)))
print('')
print('green to green:', kl(green.pdf(x), green.pdf(x)))
print('green to cyan:', kl(green.pdf(x), cyan.pdf(x)))
print('green to blue:', kl(green.pdf(x), blue.pdf(x)))
print('green to red:', kl(green.pdf(x), red.pdf(x)))
print('green to cyan0:', kl(green.pdf(x), cyan0.pdf(x)))
print('green to black:', kl(green.pdf(x), black.pdf(x)))

plt.show()
