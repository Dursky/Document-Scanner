from modules import transform
from modules import utils
from modules.document_classifier import DocumentClassifier
from modules.pdf_exporter import PDFExporter
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import cv2
import os
import argparse
import json
import datetime

class DocScanner(object):
    def __init__(self, interactive=False, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
        self.interactive = interactive
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE
        self.classifier = DocumentClassifier('document_classifier.pth')

    def get_contour(self, rescaled_image, processing_params):
        MORPH = processing_params['MORPH']
        CANNY = processing_params['CANNY']
        GAUSSIAN_BLUR = processing_params['GAUSSIAN_BLUR']

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        edged = cv2.Canny(dilated, 0, CANNY)
        test_corners = self.get_corners(edged)

        approx_contours = []
        if len(test_corners) >= 4:
            quads = []
            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype="int32")
                quads.append(points)

            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            quads = sorted(quads, key=self.angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        if not approx_contours:
            screenCnt = np.array([
                [(IM_WIDTH, 0)],
                [(IM_WIDTH, IM_HEIGHT)],
                [(0, IM_HEIGHT)],
                [(0, 0)]
            ])
        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)
            
        return screenCnt.reshape(4, 2)
    

    
    def filter_corners(self, corners, min_dist=20):
        """Filters corners that are within min_dist of others"""

        def euclidean_distance(p1, p2):
         return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
        def predicate(representatives, corner):
            return all(euclidean_distance(representative, corner) >= min_dist
                        for representative in representatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners

    def angle_range(self, quad):
        """Returns the range between max and min interior angles of quadrilateral"""
        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])
        return np.ptp([ura, ula, lra, lla])

    def get_angle(self, p1, p2, p3):
        """Returns the angle between the line segments p2->p1 and p2->p3"""
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))
        avec = a - b
        cvec = c - b
        return self.angle_between_vectors_degrees(avec, cvec)

    def angle_between_vectors_degrees(self, u, v):
        """Returns the angle between two vectors in degrees"""
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
    
    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        """Returns True if the contour satisfies all requirements"""
        return (len(cnt) == 4 and 
                cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO and 
                self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)

    def get_corners(self, img):
        """Returns a list of corners found in the input image"""
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        corners = []
        if lines is not None:
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            (contours, _) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            (contours, _) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += list(zip(corners_x, corners_y))

        corners = self.filter_corners(corners)
        return corners

    def scan(self, image_path):
        RESCALED_HEIGHT = 500.0
        OUTPUT_DIR = '../test_data/output'
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        image = cv2.imread(image_path)
        assert(image is not None)

        doc_type, confidence = self.classifier.predict(image)
        print(f"Detected document type: {doc_type} (confidence: {confidence:.2f})")
        
        processing_params = self.classifier.get_processing_params(doc_type)
        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = utils.resize(image, height=int(RESCALED_HEIGHT))
        screenCnt = self.get_contour(rescaled_image, processing_params)
        warped = transform.four_point_transform(orig, screenCnt * ratio)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sharpen = cv2.GaussianBlur(gray, processing_params['GAUSSIAN_BLUR'], 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, processing_params['ADAPTIVE_THRESH_BLOCK'], 15)

        exporter = PDFExporter(OUTPUT_DIR)
        pdf_path = exporter.save_to_pdf(image_path, thresh)
        
        output_info = {
            'document_type': doc_type,
            'confidence': float(confidence),
            'input_path': image_path,
            'output_path': pdf_path,
            'processing_params': processing_params,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        json_path = os.path.join(OUTPUT_DIR, 'output.json')
        existing_data = []
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
        if not isinstance(existing_data, list):
            existing_data = []
        existing_data.append(output_info)
        
        with open(json_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
        
        return pdf_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument("-i", action='store_true', help="Flag for manual corner verification")

    args = vars(ap.parse_args())
    scanner = DocScanner(args["i"])

    if args["image"]:
        scanner.scan(args["image"])
    else:
        im_dir = args["images"]
        valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]
        im_files = [f for f in os.listdir(im_dir) if os.path.splitext(f)[1].lower() in valid_formats]
        for im in im_files:
            scanner.scan(os.path.join(im_dir, im))