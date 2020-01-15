# Pose-Estimation---Flower
Pose Estimation is done using OpenCV ( Python )


# Getting Started
There are two python files - hsvcalib.py and final.py  
hsvcalib.py provides a toggle bar which lets you choose the HSV Color Scale for the given set of flowers.   
Once you have chosen your H (Hue), S (Saturation) and V (Value), and click the escape button, a detection window for the pose estimation of flowers appear.  
final.py finds the contours and the pose of the flower using eccentricity, angle and position of Ellipse and Rectangle Contours

# Pre-Requisites
(pip install)  
cv2  
numpy  
skimage  

# Running the tests
Just install all the dependencies mentioned above and run final.py

# Obstacles
Some cases like when the flower is facing completely away / when the flower is tilted very slightly the pose estimated fails.  If the petal is bent the hull is not deducted properly and agin the pose is not right.

