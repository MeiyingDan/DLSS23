from pattern import Checker

checkerboard = Checker(40, 4)

# draw and display the pattern
a = checkerboard.draw()
print(a)
checkerboard.show()



from pattern import Circle
circle = Circle(2000, 50, (200, 200))
circle.draw()
circle.show()




from pattern import Spectrum
spectrum = Spectrum(200)
spectrum.draw()
spectrum.show()