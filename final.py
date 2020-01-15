import cv2
import numpy as np
from skimage import morphology
from hsvcalib import ar
import math


#----------------------------------------------------------------------------------#
'''
STEP 1 : Get HSV values of flower colour using toggle bar from hsvcalib.py
         Store the HSV values in lower from array ar
'''
#----------------------------------------------------------------------------------#


lower = ar

cap = cv2.VideoCapture(0)
j=0
str=""
n_flowers=0
# blue - color for convex hull
color = (255, 0, 0)  

while True:
    _, frame = cap.read()

    #converting to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    #----------------------------------------------------------------------------------#
    '''
    STEP 2 : Masking Algorithm
            GrayScale Image
            Binary Image
            Remove noise
    '''
    #----------------------------------------------------------------------------------#
 
    lower_blue = np.array(lower)
    upper_blue = np.array([180,255,255])
    mask = cv2.inRange(hsv,lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask = mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,9)

    # Return a new array of given shape and type, filled with ones. 
    # np.ones(shape, datatype - optional, order='C' - rowmajor or column major - optional)
    kernel1 = np.ones((17, 17), np.uint8)
    kernel = np.ones((5,5),np.uint8)

    ret, bin_ = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # Performs advanced morphological transformations.
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kernel1)
    bin_ = cv2.dilate(bin_, kernel, iterations=1)

    # skikit- image : Remove objects smaller than the specified size.
    bin_ = morphology.remove_small_objects(bin_, min_size=500,  in_place=True)

    bin_ = cv2.erode(bin_, kernel, iterations=2)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, kernel)


    # remove small contours less than 500 in size to ensure the flower has only one hull 
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 700  
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    # Scales, calculates absolute values, and converts the result to 8-bit.
    img2 = cv2.convertScaleAbs(img2)
    cv2.waitKey(10)

    #----------------------------------------------------------------------------------#
    '''
    STEP 3 : Contours
            Hull Array for Convex Hull Points
            Points for each Contour
            Draw Contour
    '''
    #----------------------------------------------------------------------------------#

    # Finds contours in a binary image
    contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # create hull array for convex hull points
    hull = []
    hullpoints = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour and strong in the correct format 
        if(len(contours) == 1):
            # to avoid extra brackets
            hull.append(list(cv2.convexHull(contours[i], False)))
            hullpoints.append(list(cv2.convexHull(contours[i], False)))
        else:
            hull.append(cv2.convexHull(contours[i], False))
            hullpoints.append(cv2.convexHull(contours[i], False))

    hullf = np.array(hullpoints)
    hullf = list(hullf) 
                
    for i in range (len(contours)):
        cv2.drawContours(frame, contours, i, color, 1,8,hierarchy)
        cv2.drawContours(frame, np.array(hull), i, color, 1, 8)

    #----------------------------------------------------------------------------------#
    '''
    STEP 4 : Ellipse - cv2.fitEllipse()
            Find angle, lengths of axis and centre of Ellipse using Contours
            Calculate Eccentricity of Ellipse
    '''
    #----------------------------------------------------------------------------------#

    for i in range(len(contours)):
        # draw an ellipse for the contour
        ellipse = cv2.fitEllipse(contours[i])
        cv2.ellipse(frame,ellipse,(255,0,0),2)

        # (x,y) stands for centre of ellipse, (MA, ma) stands for length of Major and Minor axis, angle is 
        # the angle of ellipse
        (x,y),(MA,ma),angle = cv2.fitEllipse(contours[i])
        a = ma/2
        b = MA/2

        # eccentricity of an ellipse will be 1 if it is closer to a line, 
        # 0 if it is closer to a circle
        eccentricity = math.sqrt(math.pow(a, 2)-math.pow(b, 2))
        eccentricity = round(eccentricity/a, 2)


        print("Eccentricity : %.2f" %eccentricity)

        print("\nAngle of Ellipse : %d\n" %angle)

        #----------------------------------------------------------------------------------#
        '''
        STEP 5 : Rectangle - cv2.minAreaRect()
                cv2.boxPoints() - Points of Rectangle -> stored in box
                Draw Rectangle Contours
                Find angle of Rectangle
        '''
        #----------------------------------------------------------------------------------#

        # Find the rotated rectangles for each contour
        minRect = [None]*len(contours)
        for i, c in enumerate(contours):
            minRect[i] = cv2.minAreaRect(c)

        # Draw rotated rectangles
        for i, c in enumerate(contours):
            # rotated rectangle
            box = cv2.boxPoints(minRect[i])
            box = np.intp(box) 
            cv2.drawContours(frame, [box], 0, color)

        # similar to fit ellipse, rect[2] will represent angle of rectangle
        rect = cv2.minAreaRect(contours[0])
        rangle = 0 - rect[2]
        print("Angle of Rectangle : %d\n" %rangle)

        #----------------------------------------------------------------------------------#
        '''
        STEP 6 : Position of Flower
                Using Eccentricity of Ellipse
                Using Angle of Ellipse 
                Using Coordinates of Rectangle
        '''
        #----------------------------------------------------------------------------------#

        # if ecc is < 0.5 it will be closer to a circle and hence facing the camera
        if(eccentricity <0.5):
            eccs = 'Towards camera'
        elif(eccentricity > 0.8):
            if(angle < 70):
                eccs = 'Fully Right/ Left'
            else:
                if(box[0][0] > max(box[0])/2):
                    eccs = 'Fully Up/ Down'
                else:
                    eccs = 'Fully Right/ Left'

        else:
            if(angle > 75):
                eccs = 'Tilted Up/ Down'
            else:
                eccs = 'Tilted Right/ Left'


        cv2.putText(frame, eccs, (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))

    j=j+1
    if j>20 :
        n_flowers = len(contours)
        str = "Number of flower(s) : %s" % n_flowers
        j=1
        

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, 'Scanning for flowers...', (10, 400), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Detect",frame)
    j=j+1


    k = cv2.waitKey(5) & 0xFF
    
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()

'''
#-----------------------------------------------------------------------------------#
cv2 Functions : -
1. inRange() - Checks if array elements lie between the elements of two other arrays.
            cv2.inRange(first input array, inclusive lower boundary array, upperb – inclusive upper boundary array)
2. bitwise_and() - Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar.
                 cv2.bitwise_and(first input array, second input array, optional operation mask that specifies
                 elements of the output array to be changed)
3. cvtColor() -  Converts an image from one color space to another.
                cv2.cvtColor(input image, color space conversion code)
4. medianBlur() - Blurs an image using the median filter.
5. threshold() - The function is typically used to get a binary image out of a grayscale image or for removing 
                a noise, that is, filtering out pixels with too small or too large values.
                cv2.threshold(src, threshold value, max value, type)
6. dilate() - Dilates an image by using a specific structuring element. The function dilates the source image using 
            the specified structuring element that determines the shape of a pixel neighborhood over which the maximum is taken
7. erode() -  What this does is to compute a local minimum over the area of the kernel.
8. connectedComponentsWithStats() - Computes the connected components labeled image of boolean image and also produces 
                                a statistics output for each label
#-----------------------------------------------------------------------------------#
'''
