import cv2
import numpy as np
import sys
import time

def energy(img):
    gaus_blurr = cv2.GaussianBlur(img, (3, 3), 0, 0)
    img_gray = cv2.cvtColor(gaus_blurr, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT, )
    dy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT, )

    return cv2.add(np.absolute(dx), np.absolute(dy))

def create_mask(image):
    global pts; pts = []
    global x_lst; x_lst = []
    global img_m; img_m = image.copy()
    global mask_m; mask_m = np.zeros(img_m.shape[:2])

    def draw_mask(pts):
        pts = np.array(pts, dtype=np.int32)
        pts.reshape((-1,1,2))
        cv2.fillPoly(mask_m,[pts],(255))
        cv2.imshow("image",mask_m)
        cv2.imwrite("mask.jpg", mask_m)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mousePoints(event, x,y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            img_1 = cv2.circle(img_m, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imshow("original", img_1)
            pts.append((x,y))
            x_lst.append(x)
        elif event == cv2.EVENT_RBUTTONDOWN:
            draw_mask(pts)

    cv2.imshow("original", img_m)
    cv2.setMouseCallback("original", mousePoints)
    cv2.waitKey(0)
    return mask_m, (max(x_lst) - min(x_lst))

def cumulative_vertical_energies(energy):

    energy_map = np.zeros(energy.shape)

    for i in range(1, energy.shape[0]):
        for j in range(energy.shape[1]):
            if energy.shape[1] -1 > j >= 1 :
                left = energy_map[i - 1, j - 1]
                right = energy_map[i - 1, j + 1]
                energy_map[i, j] = energy[i, j] + min(left, energy_map[i - 1, j], right)
            elif j < 1 :
                right = energy_map[i - 1, j + 1]
                energy_map[i, j] = energy[i, j] + min(energy_map[i - 1, j], right)
            else:
                left = energy_map[i - 1, j - 1]
                energy_map[i, j] = energy[i, j] + min(left, energy_map[i - 1, j])

    return energy_map

def cumulative_horizontal_energies(energy):

    energy_map = np.zeros(energy.shape)

    for j in range(1, energy.shape[1]):
        for i in range(energy.shape[0]):
            if energy.shape[0] -1 > i >= 1 :
                top = energy_map[i - 1, j - 1]
                bottom = energy_map[i + 1, j - 1]
                energy_map[i, j] = energy[i, j] + min(top, energy_map[i , j - 1], bottom)
            elif i < 1 :
                bottom = energy_map[i + 1, j - 1]
                energy_map[i, j] = energy[i, j] + min(energy_map[i , j - 1], bottom)
            else:
                top = energy_map[i - 1, j - 1]
                energy_map[i, j] = energy[i, j] + min(top, energy_map[i , j - 1])

    return energy_map

global seam_2;  seam_2 = []
def horizontal_seam(energy_map):
    height, width = energy_map.shape[0], energy_map.shape[1]
    prev = 0
    seam_2.clear()

    return horizon_seam(energy_map, width-1, height, width, prev)

def horizon_seam(energy_map, i, height, width , prev):
    if i == -1:
        return seam_2
    
    col = energy_map[:, i]
    if i == width - 1:
        prev = np.argmin(col)
    else:
        if height - 1 > prev >= 1:
            top = col[prev - 1]
            bottom = col[prev + 1]
            if min(top, col[prev], bottom) == top:
                prev += -1
            elif min(top, col[prev], bottom) == col[prev]:
                prev += 0
            else:
                prev += 1
        elif prev < 1:
            bottom = col[prev + 1]
            if min(col[prev], bottom) == col[prev]:
                prev += 0
            else:
                prev += 1
        else:
            top = col[prev - 1]
            if min(top, col[prev]) == top:
                prev += -1
            else:
                prev += 0

    seam_2.append([i, prev])
    return horizon_seam(energy_map, i-1, height, width, prev)

global seam_1;  seam_1 = []
def vertical_seam(energy_map):
    height, width = energy_map.shape[0], energy_map.shape[1]
    prev = 0
    seam_1.clear()

    return verti_seam(energy_map, height-1, height, width, prev)

def verti_seam(energy_map, i, height, width , prev):
    if i == -1:
        return seam_1
    
    row = energy_map[i, :]
    if i == height - 1:
        prev = np.argmin(row)
    else:
        if width - 1 > prev >= 1:
            left = row[prev - 1]
            right = row[prev + 1]
            if min(left, row[prev], right) == left:
                prev += -1
            elif min(left, row[prev], right) == row[prev]:
                prev += 0
            else:
                prev += 1
        elif prev < 1:
            right = row[prev + 1]
            if min(row[prev], right) == row[prev]:
                prev += 0
            else:
                prev += 1
        else:
            left = row[prev - 1]
            if min(left, row[prev]) == left:
                prev += -1
            else:
                prev += 0
    seam_1.append([prev, i])
    return verti_seam(energy_map, i-1, height, width, prev)

def remove_horizontal_seam(img, seam):
    removed = np.zeros((img.shape[0] - 1, img.shape[1], img.shape[2]), np.uint8)

    for v in reversed(seam):
        removed[0:v[1], v[0]] = img[0:v[1], v[0]]
        removed[v[1]:img.shape[0] - 1, v[0]] = img[v[1] + 1:img.shape[0], v[0]]

    return removed

def remove_vertical_seam(img, seam):
    removed = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    
    for v in reversed(seam):
        removed[v[1], 0:v[0]] = img[v[1], 0:v[0]]
        removed[v[1], v[0]:img.shape[1] - 1] = img[v[1], v[0] + 1:img.shape[1]]

    return removed

def normal_seam_carving(img, start_time, width):

    ratio = img.shape[1] / width
    height = int(img.shape[0] / ratio)

    cv2.namedWindow('seam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('seam', 300, 300)
    cv2.imwrite("Initial.jpg", img)

    dx = img.shape[1] - width if img.shape[1] > width else 0
    dy = img.shape[0] - height if img.shape[0] > height else 0

    for i in range(dy):
        energy_map = cumulative_horizontal_energies(energy(img))
        seam = horizontal_seam(energy_map)
        cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 0, 255))
        cv2.imshow('seam', img)
        cv2.waitKey(1)
        img = remove_horizontal_seam(img, seam)

    for i in range(dx):
        energy_map = cumulative_vertical_energies(energy(img))
        seam = vertical_seam(energy_map)
        cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 0, 255))
        cv2.imshow('seam', img)
        cv2.waitKey(1)
        img = remove_vertical_seam(img, seam)

    
    print("Time of execution: ", time.time() - start_time)
    cv2.imwrite('Result.jpg', img)
    print("\nSeams has been removed successfully")
    print("\nFinal Dimensions: ",img.shape[1]," x ",img.shape[0]," x ",img.shape[2])

    cv2.imshow('seam', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def object_removal(img, mask, dx, start_time):

    cv2.namedWindow('seam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('seam', 300, 300)

    for i in range(dx):
        var = energy(img)
        # cv2.imshow("energy_function", var)
        # cv2.waitKey(1)
        for u in range(var.shape[0]):
            for v in range(var.shape[1]):
                if mask[u][v] > 0:
                    var[u][v] = -1000
        energy_map = cumulative_vertical_energies(var)
        seam = vertical_seam(energy_map)
        cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 0, 255))
        cv2.imshow('seam', img)
        cv2.waitKey(1)
        img = remove_vertical_seam(img, seam)

    print("Time of execution: ", time.time() - start_time)
    cv2.imwrite('Result.jpg', img)
    print("\nObject has been removed successfully")
    print("\nFinal Dimensions: ",img.shape[1]," x ",img.shape[0]," x ",img.shape[2])

    cv2.imshow('seam', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def guassian_pyramid(img, start_time, width):

    lyr_1 = cv2.pyrDown(img)
    print("\nDimensions of Layer 1 of Pyramid: " ,lyr_1.shape[1]," x ",lyr_1.shape[0]," x ",lyr_1.shape[2])

    ratio = img.shape[1] / width
    height = int(img.shape[0] / ratio)

    cv2.namedWindow('seam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('seam', 300, 300)
    cv2.imwrite("Initial.jpg", img)

    dx = img.shape[1] - width if img.shape[1] > width else 0
    dy = img.shape[0] - height if img.shape[0] > height else 0

    dx = dx//2;    dy = dy//2
    seam_pyr_1 = [];    seam_pyr_2 = []

    for i in range(dy):
        energy_map = cumulative_horizontal_energies(energy(lyr_1))
        seam = horizontal_seam(energy_map)
        x = img.copy()
        seam_pyr_1.clear(); seam_pyr_2.clear()

        for i in seam:
            seam_pyr_1.append([i[0]*2,i[1]*2])
            seam_pyr_1.append([i[0]*2+1,i[1]*2])
            
            seam_pyr_2.append([i[0]*2,i[1]*2+1])
            seam_pyr_2.append([i[0]*2+1,i[1]*2+1])

        cv2.polylines(img, np.int32([np.asarray(seam_pyr_1)]), False, (0, 255, 0))
        cv2.polylines(img, np.int32([np.asarray(seam_pyr_2)]), False, (0, 0, 255))
        cv2.imshow('seam', img)
        img = x.copy()
        cv2.waitKey(1)

        lyr_1 = remove_horizontal_seam(lyr_1, seam)
        img = remove_horizontal_seam(img, seam_pyr_2)
        img = remove_horizontal_seam(img, seam_pyr_1)

    for i in range(dx):
        energy_map = cumulative_vertical_energies(energy(lyr_1))
        seam = vertical_seam(energy_map)
        x = img.copy()
        seam_pyr_1.clear(); seam_pyr_2.clear()

        for i in seam:
            seam_pyr_1.append([i[0]*2,i[1]*2])
            seam_pyr_1.append([i[0]*2,i[1]*2+1])

            seam_pyr_2.append([i[0]*2+1,i[1]*2])
            seam_pyr_2.append([i[0]*2+1,i[1]*2+1])

        cv2.polylines(img, np.int32([np.asarray(seam_pyr_1)]), False, (0, 255, 0))
        cv2.polylines(img, np.int32([np.asarray(seam_pyr_2)]), False, (0, 0, 255))
        cv2.imshow('seam', img)
        img = x.copy()
        cv2.waitKey(1) 

        lyr_1=remove_vertical_seam(lyr_1, seam)
        img = remove_vertical_seam(img, seam_pyr_2)
        img = remove_vertical_seam(img, seam_pyr_1)
     
    print("Time of execution: ", time.time() - start_time)
    cv2.imwrite('Result.jpg', img)
    print("\nSeams has been removed successfully")
    print("\nFinal Dimensions: ",img.shape[1]," x ",img.shape[0]," x ",img.shape[2])

    cv2.imshow('seam', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread(sys.argv[1])

h = int((img.shape[0] * 500) / img.shape[1])
h = h+1 if h%2 != 0 else h
img = cv2.resize(img, (500, h))

while True:
    op = int(input("\nEnter 1 for Normal Seam Carving\nEnter 2 for Object Removal\nEnter 3 for Seam Carving using Guassian pyramid: "))
    if op == 1:
        print("Initial Dimensions: ",img.shape[1]," x ",img.shape[0]," x ",img.shape[2])
        width = int(input('\nEnter Width of the Resized Image: '))
        start_time = time.time()
        normal_seam_carving(img, start_time, width)
        break
    elif op == 2:
        print("Initial Dimensions: ",img.shape[1]," x ",img.shape[0]," x ",img.shape[2])
        mask, dx = create_mask(img)
        start_time = time.time()
        object_removal(img, mask, dx, start_time)
        break
    elif op == 3:
        print("Initial Dimensions: ",img.shape[1]," x ",img.shape[0]," x ",img.shape[2])
        width = int(input('\nEnter Width of the Resized Image: '))
        start_time = time.time()
        guassian_pyramid(img, start_time, width)
        break
    else:
        print("Please enter valid input...")