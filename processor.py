import cv2
import math
import numpy as np

vidsize = (640,480,3)

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    
    # Retrieve the number of color channels of the image.
    #channel_count = img.shape[2]
    
    # color used to fill polygon
    match_mask_color = 255
      
    # Fill the polygon with white
    cv2.fillPoly(mask, vertices, (255,255,255))
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image

def gamma_correction(RGBimage, correct_param = 0.35,equalizeHist = False):
    red = RGBimage[:,:,2]
    green = RGBimage[:,:,1]
    blue = RGBimage[:,:,0]
    
    red = red/255.0
    red = cv2.pow(red, correct_param)
    red = np.uint8(red*255)
    if equalizeHist:
        red = cv2.equalizeHist(red)
    
    green = green/255.0
    green = cv2.pow(green, correct_param)
    green = np.uint8(green*255)
    if equalizeHist:
        green = cv2.equalizeHist(green)
        
    
    blue = blue/255.0
    blue = cv2.pow(blue, correct_param)
    blue = np.uint8(blue*255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)
    

    output = cv2.merge((blue,green,red))
    return output

def gamma_correction_auto(RGBimage,equalizeHist = False): #0.35
    originalFile = RGBimage.copy()
    red = RGBimage[:,:,2]
    green = RGBimage[:,:,1]
    blue = RGBimage[:,:,0]
    
    forLuminance = cv2.cvtColor(originalFile,cv2.COLOR_BGR2YUV)
    Y = forLuminance[:,:,0]
    totalPix = vidsize[0]* vidsize[1]
    summ = np.sum(Y[:,:])
    Yaverage = np.divide(totalPix,summ)
    #Yclipped = np.clip(Yaverage,0,1)
    epsilon = 1.19209e-007
    correct_param = np.divide(-0.3,np.log10([Yaverage + epsilon]))
    correct_param = 0.7 - correct_param 

    red = red/255.0
    red = cv2.pow(red, correct_param)
    red = np.uint8(red*255)
    if equalizeHist:
        red = cv2.equalizeHist(red)
    
    green = green/255.0
    green = cv2.pow(green, correct_param)
    green = np.uint8(green*255)
    if equalizeHist:
        green = cv2.equalizeHist(green)
        
    
    blue = blue/255.0
    blue = cv2.pow(blue, correct_param)
    blue = np.uint8(blue*255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)
    

    output = cv2.merge((blue,green,red))
    #print(correct_param)
    return output

def hsv_filter(image, min_val_y, max_val_y,  min_val_w, max_val_w):
    """
    A function returning a mask for pixels within min_val - max_val range
    Inputs:
    - image - a BGR image you want to apply function on
    - min_val_y - array of shape (3,) giving minumum HSV values for yellow color
    - max_val_y - array of shape (3,) giving maximum HSV values for yellow color
    - min_val_w - array of shape (3,) giving minumum HSV values for white color
    - max_val_w - array of shape (3,) giving maximum HSV values for white color
    Returns:
    - img_filtered - image of pixels being in given threshold
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, min_val_y, max_val_y)
    mask_white = cv2.inRange(hsv, min_val_w, max_val_w)
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    img_filtered = cv2.bitwise_and(image, image, mask=mask)
    
    return img_filtered

def hough_transform(original, gray_img, threshold, discard_horizontal = 0.4):
    """
    A function fitting lines that intersect >=threshold white pixels
    Input:
    - original - image we want to draw lines on
    - gray_img - image with white/black pixels, e.g. a result of Canny Edge Detection
    - threshold - if a line intersects more than threshold white pixels, draw it
    - discard_horizontal - smallest abs derivative of line that we want to take into account
    Return:
    - image_lines - result of applying the function
    - lines_ok - rho and theta
    """
    lines = cv2.HoughLines(gray_img, 0.5, np.pi / 360, threshold)
    image_lines = original
    lines_ok = [] #list of parameters of lines that we want to take into account (not horizontal)
            
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            #discard horizontal lines
            m = -math.cos(theta)/(math.sin(theta)+1e-10) #adding some small value to avoid dividing by 0
            if abs(m) < discard_horizontal:
                continue
            else:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(image_lines, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
                lines_ok.append([rho,theta])
        
    lines_ok = np.array(lines_ok)
                    
    return image_lines, lines_ok

def clustering(lines, original, region_of_interest_points, eps = 0.05, min_samples = 3):
    """
    A function using DBSCAN clustering algorithm for finding best lines to be drawn on the output video
    Inputs:
    - lines - output of hough tranform function, array containing parameters of found lines
    - original - image we want to draw final lines on
    - region_of_interest_points - for drawing lines of desired length
    Output:
    - img - image with detected lane lines drawn
    """
    img = original
    img_lines = np.zeros_like(img, dtype=np.int32)

    if lines.shape[0] != 0:
        #preprocessing features to be in (0-1) range
        scaler = MinMaxScaler()
        scaler.fit(lines)
        lines = scaler.fit_transform(lines)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(lines) #applying DBSCAN Algorithm on our normalized lines
        labels = db.labels_

        lines = scaler.inverse_transform(lines) #getting back our original values

        grouped = defaultdict(list)
        #grouping lines by clusters
        for i, label in enumerate(labels):
            grouped[label].append([lines[i,0],lines[i,1]])

        num_clusters = np.max(labels) + 1
        means = []
        #getting mean values by cluster
        for i in range(num_clusters):
            mean = np.mean(np.array(grouped[i]), axis=0)
            means.append(mean)

        means = np.array(means)
        
        #printing the result on original image
        for rho, theta in means:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (255,255,255), 2, cv2.LINE_AA)
        
    return img

def IPM(image,ROI_points):
    pts_src = np.array(ROI_points,dtype=float)
    size = (700,600,3)
    ipm_out = np.zeros(size, np.uint8)
    
    dst_points = np.array(
                       [
                        [0,0],
                        [size[0] - 1, 0],
                        [size[0] - 1, size[1] -1],
                        [0, size[1] - 1 ]
                        ], dtype=float
                       )
    h, status = cv2.findHomography(pts_src, dst_points)
    ipm_out = cv2.warpPerspective(image, h, size[0:2])
    ipm_out = cv2.rotate(ipm_out,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return ipm_out