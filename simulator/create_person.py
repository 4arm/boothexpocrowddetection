import cv2
import numpy as np

# Create a transparent 200x500 image
img = np.zeros((500, 200, 4), dtype=np.uint8)

# Colors (Maroon shirt, blue jeans, skin color)
skin = (170, 210, 255, 255) # BGR + Alpha
maroon = (40, 20, 110, 255)
blue = (150, 80, 20, 255)
black = (0, 0, 0, 255)

# Head
cv2.ellipse(img, (100, 50), (35, 45), 0, 0, 360, skin, -1)
# Body (Torso)
cv2.ellipse(img, (100, 200), (70, 110), 0, 0, 360, maroon, -1)
# Arms
cv2.ellipse(img, (30, 210), (20, 90), 10, 0, 360, maroon, -1)
cv2.ellipse(img, (170, 210), (20, 90), -10, 0, 360, maroon, -1)
# Hands
cv2.circle(img, (20, 290), 15, skin, -1)
cv2.circle(img, (180, 290), 15, skin, -1)
# Legs
cv2.rectangle(img, (60, 300), (95, 480), blue, -1)
cv2.rectangle(img, (105, 300), (140, 480), blue, -1)
# Shoes
cv2.ellipse(img, (75, 490), (25, 15), 0, 0, 360, black, -1)
cv2.ellipse(img, (125, 490), (25, 15), 0, 0, 360, black, -1)

cv2.imwrite('/home/pi/booth/simulator/static/person.png', img)
