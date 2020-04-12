# Read and process layout image
import cv2
import numpy as np
import os

i = 0
for filename in os.listdir("./originals"):
    layout = cv2.imread("originals/" + filename)

    # Increase contrast
    layout_hsv = cv2.cvtColor(layout, cv2.COLOR_BGR2HSV).astype('float32')
    layout_hsv[:,:,2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in layout_hsv[:,:,2]]
    layout = cv2.cvtColor(layout_hsv.astype('uint8'), cv2.COLOR_HSV2BGR)

    # Grayscale
    layout_gray = cv2.cvtColor(layout, cv2.COLOR_BGR2GRAY)
    ret, layout_thresh = cv2.threshold(layout_gray, 127, 255, cv2.THRESH_BINARY_INV)
    layout_thresh = 255 - layout_thresh

    # Get card contours
    layout_thresh, card_contours, hierarchy = cv2.findContours(layout_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(layout, card_contours, -1, (0, 255, 0))
    # cv2.imwrite('test.jpg', layout)

    # Specify dimensions of result rectangle
    w = 200
    h = 200

    # Result rectangle corners, Order to match approx: top left, bottom left, bottom right, top right
    rect = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], np.float32)

    # Filter card candidates, warp and classify each card
    for card in card_contours:
        # Get contour and perimeter length, check perim for faulty contours
        perim = cv2.arcLength(card, True)
        # Ignore random contours in the layout image that aren't cards
        if perim < 100:
            continue
        # Get approx corners of card
        approx = cv2.approxPolyDP(card, 0.02*perim, True)
        # Reshape approx to match rect
        approx = np.squeeze(approx).astype(np.float32)
        # Warp
        try:
            transform = cv2.getPerspectiveTransform(approx, rect)
        except:
            continue
        warp = cv2.warpPerspective(layout, transform, (w, h))

        cv2.imwrite(str(i) + '.jpg', warp)
        i += 1