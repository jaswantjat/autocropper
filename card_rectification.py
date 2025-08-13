"""
Card Rectification Module
Modified version of the original shakex/card-rectification algorithm
Designed to work as an importable module for Flask API integration

Original author: kxie
Modified for API integration
"""

import os
import cv2
import torch
import imutils
import numpy as np
import logging
from skimage import exposure, img_as_ubyte
from imutils.perspective import four_point_transform
from itertools import combinations
from torchvision import transforms

# Configure module logger
logger = logging.getLogger(__name__)

# Processing configuration
PROCESS_SIZE = 1000
MODEL_INPUT_SIZE = 1000

class CardRectificationError(Exception):
    """Custom exception for card rectification errors."""
    pass

class EdgeDetectionError(CardRectificationError):
    """Exception raised when edge detection fails."""
    pass

class CornerDetectionError(CardRectificationError):
    """Exception raised when corner detection fails."""
    pass

class TransformationError(CardRectificationError):
    """Exception raised when perspective transformation fails."""
    pass

def get_card_colormap():
    """Get colormap for edge detection visualization."""
    return np.asarray([[0, 0, 0], [255, 255, 255]])

def decode_map(label_mask):
    """Decode the neural network output to RGB image."""
    try:
        label_colors = get_card_colormap()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, 2):
            r[label_mask == ll] = label_colors[ll, 0]
            g[label_mask == ll] = label_colors[ll, 1]
            b[label_mask == ll] = label_colors[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb.astype(np.uint8)
    except Exception as e:
        logger.error(f"Failed to decode map: {e}")
        raise EdgeDetectionError(f"Map decoding failed: {e}")

def detect_edge_cnn(img, model, device):
    """
    CNN-based edge detection using the trained neural network.
    
    Args:
        img: Input image as numpy array
        model: Trained PyTorch model
        device: PyTorch device (cuda/cpu)
        
    Returns:
        Edge detected image as numpy array
        
    Raises:
        EdgeDetectionError: If CNN edge detection fails
    """
    try:
        logger.debug("Starting CNN edge detection...")
        
        # Resize image for model input
        image = cv2.resize(img, (MODEL_INPUT_SIZE, int(MODEL_INPUT_SIZE * img.shape[0] / img.shape[1])),
                           interpolation=cv2.INTER_LINEAR)
        
        # Prepare image for neural network
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.196, 0.179, 0.323], [0.257, 0.257, 0.401])
        ])
        image = tf(image)
        image = image.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            img_val = image.to(device)
            res = model(img_val)
            pred = np.squeeze(res.data.max(1)[1].cpu().numpy())
            edged = decode_map(pred)
            edged = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
            edged = cv2.resize(edged, (PROCESS_SIZE, int(PROCESS_SIZE * img.shape[0] / img.shape[1])),
                               interpolation=cv2.INTER_NEAREST)
        
        logger.debug("CNN edge detection completed successfully")
        return edged
        
    except Exception as e:
        logger.error(f"CNN edge detection failed: {e}")
        raise EdgeDetectionError(f"CNN edge detection failed: {e}")

def detect_edge_traditional(img):
    """
    Traditional edge detection using image processing techniques.
    
    Args:
        img: Input image as numpy array
        
    Returns:
        Edge detected image as numpy array
        
    Raises:
        EdgeDetectionError: If traditional edge detection fails
    """
    try:
        logger.debug("Starting traditional edge detection...")
        
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Adjust for lighting conditions
        mean_gray = cv2.mean(gray)
        TH_LIGHT = 150
        if mean_gray[0] > TH_LIGHT:
            gray = exposure.adjust_gamma(gray, gamma=6)
            gray = exposure.equalize_adapthist(gray, kernel_size=None, clip_limit=0.02)
            gray = img_as_ubyte(gray)

        # Apply morphological operations and filtering
        kernel = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        blurred = cv2.medianBlur(closing, 5)
        blurred = cv2.bilateralFilter(blurred, d=0, sigmaColor=15, sigmaSpace=10)
        
        # Edge detection
        edged = cv2.Canny(blurred, 75, 200)
        
        logger.debug("Traditional edge detection completed successfully")
        return edged
        
    except Exception as e:
        logger.error(f"Traditional edge detection failed: {e}")
        raise EdgeDetectionError(f"Traditional edge detection failed: {e}")

def cross_point(line1, line2):
    """
    Calculate intersection point of two lines.
    
    Args:
        line1: First line as [x1, y1, x2, y2]
        line2: Second line as [x3, y3, x4, y4]
        
    Returns:
        Intersection point as [x, y]
    """
    try:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate slopes and intercepts
        if (x2 - x1) == 0:
            k1 = None
        else:
            k1 = (y2 - y1) * 1.0 / (x2 - x1)
            b1 = y1 * 1.0 - x1 * k1 * 1.0
        
        if (x4 - x3) == 0:
            k2 = None
            b2 = 0
        else:
            k2 = (y4 - y3) * 1.0 / (x4 - x3)
            b2 = y3 * 1.0 - x3 * k2 * 1.0
        
        # Find intersection
        x, y = 0, 0
        if k1 is None:
            if k2 is not None:
                x = x1
                y = k2 * x1 + b2
        elif k2 is None:
            x = x3
            y = k1 * x3 + b1
        elif k2 != k1:
            x = (b2 - b1) * 1.0 / (k1 - k2)
            y = k1 * x * 1.0 + b1 * 1.0
        
        return [x, y]
        
    except Exception as e:
        logger.warning(f"Cross point calculation failed: {e}")
        return [0, 0]

def get_angle(sta_point, mid_point, end_point):
    """
    Calculate angle between three points.
    
    Args:
        sta_point: Starting point
        mid_point: Middle point (vertex)
        end_point: End point
        
    Returns:
        Angle in degrees
    """
    try:
        ma_x = sta_point[0][0] - mid_point[0][0]
        ma_y = sta_point[0][1] - mid_point[0][1]
        mb_x = end_point[0][0] - mid_point[0][0]
        mb_y = end_point[0][1] - mid_point[0][1]
        ab_x = sta_point[0][0] - end_point[0][0]
        ab_y = sta_point[0][1] - end_point[0][1]
        ab_val2 = ab_x * ab_x + ab_y * ab_y
        ma_val2 = ma_x * ma_x + ma_y * ma_y
        mb_val2 = mb_x * mb_x + mb_y * mb_y
        
        # Avoid division by zero
        denominator = 2 * np.sqrt(ma_val2) * np.sqrt(mb_val2)
        if denominator == 0:
            return 0
            
        cos_M = (ma_val2 + mb_val2 - ab_val2) / denominator
        # Clamp to valid range for arccos
        cos_M = np.clip(cos_M, -1.0, 1.0)
        angleAMB = np.arccos(cos_M) / np.pi * 180
        return angleAMB
    except Exception as e:
        logger.warning(f"Angle calculation failed: {e}")
        return 0

def validate_corners(approx):
    """
    Validate that detected corners form a valid quadrilateral.
    
    Args:
        approx: Detected corner points
        
    Returns:
        True if valid
        
    Raises:
        CornerDetectionError: If corners are invalid
    """
    try:
        hull = cv2.convexHull(approx)
        TH_ANGLE = 45
        
        if len(hull) != 4:
            raise CornerDetectionError(f"Expected 4 corners, found {len(hull)}")
        
        for i in range(4):
            p1 = hull[(i - 1) % 4]
            p2 = hull[i]
            p3 = hull[(i + 1) % 4]
            angle = get_angle(p1, p2, p3)
            
            if not (90 - TH_ANGLE < angle < 90 + TH_ANGLE):
                raise CornerDetectionError(f"Invalid corner angle: {angle:.1f}°")
        
        logger.debug("Corner validation passed")
        return True
        
    except CornerDetectionError:
        raise
    except Exception as e:
        logger.error(f"Corner validation failed: {e}")
        raise CornerDetectionError(f"Corner validation failed: {e}")

def _detect_corners_by_contour(img, edged):
    """
    Fallback corner detection using contour approximation.

    Args:
        img: Original image
        edged: Edge-detected image

    Returns:
        numpy.ndarray: Array of 4 corner points

    Raises:
        CornerDetectionError: If corner detection fails
    """
    logger.debug("Starting contour-based corner detection")

    try:
        # Find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2(or_better=True) else cnts[1]

        if not cnts:
            raise CornerDetectionError("No contours found for fallback detection")

        # Sort contours by area and try the largest ones
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Calculate image area once
        img_area = img.shape[0] * img.shape[1]

        for i, contour in enumerate(cnts[:5]):  # Try top 5 contours
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            logger.debug(f"Contour {i}: area={cv2.contourArea(contour):.0f}, vertices={len(approx)}")

            # If we have 4 vertices, this might be our card
            if len(approx) == 4:
                # Check if the contour is large enough
                area = cv2.contourArea(approx)

                if area > 0.1 * img_area:  # At least 10% of image area
                    logger.debug(f"Found potential card contour with area {area:.0f}")

                    # Reshape to expected format
                    corners = approx.reshape(4, 2).astype(np.float32)

                    # Order corners: top-left, top-right, bottom-right, bottom-left
                    corners = _order_corners(corners)

                    # Validate corners
                    try:
                        validate_corners(corners.reshape(4, 1, 2))
                        logger.debug("Contour-based corner detection successful")
                        return corners.reshape(4, 1, 2)
                    except CornerDetectionError as e:
                        logger.debug(f"Corner validation failed: {e}")
                        continue

            # Try with different epsilon values for approximation
            for eps_factor in [0.01, 0.03, 0.05]:
                epsilon = eps_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    if area > 0.05 * img_area:  # Lower threshold
                        corners = approx.reshape(4, 2).astype(np.float32)
                        corners = _order_corners(corners)

                        try:
                            validate_corners(corners.reshape(4, 1, 2))
                            logger.debug(f"Contour-based detection successful with epsilon {eps_factor}")
                            return corners.reshape(4, 1, 2)
                        except CornerDetectionError:
                            continue

        raise CornerDetectionError("No valid rectangular contour found")

    except Exception as e:
        logger.error(f"Contour-based corner detection failed: {e}")
        raise CornerDetectionError(f"Contour-based corner detection failed: {e}")

def _order_corners(corners):
    """Order corners in clockwise order starting from top-left."""
    # Calculate center point
    center = np.mean(corners, axis=0)

    # Sort by angle from center
    def angle_from_center(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])

    # Sort corners by angle
    sorted_corners = sorted(corners, key=angle_from_center)

    # Find top-left corner (smallest x + y)
    sums = [pt[0] + pt[1] for pt in sorted_corners]
    top_left_idx = np.argmin(sums)

    # Reorder starting from top-left
    ordered = []
    for i in range(4):
        ordered.append(sorted_corners[(top_left_idx + i) % 4])

    return np.array(ordered, dtype=np.float32)

def detect_corners(edged, img, ratio):
    """
    Detect four corners of the ID card from edge image.

    Args:
        edged: Edge detected image
        img: Original resized image
        ratio: Scaling ratio

    Returns:
        Corner points scaled to original image size

    Raises:
        CornerDetectionError: If corner detection fails
    """
    try:
        logger.debug("Starting corner detection...")

        # Dilate edges and create mask
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)
        mask = np.zeros((edged.shape[0], edged.shape[1]), np.uint8)
        mask[10:edged.shape[0] - 10, 10:edged.shape[1] - 10] = 1
        edged = edged * mask

        # Find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if imutils.is_cv2(or_better=True) else cnts[1]

        if not cnts:
            raise CornerDetectionError("No contours found")

        cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)
        edgelines = np.zeros(edged.shape, np.uint8)
        cNum = 4

        # Process contours to create edge lines with improved logic
        logger.debug(f"Found {len(cnts)} contours")

        for i in range(min(cNum, len(cnts))):
            contour_area = cv2.contourArea(cnts[i])
            TH = 1 / 20.0
            area_threshold = TH * img.shape[0] * img.shape[1]

            logger.debug(f"Contour {i}: area={contour_area:.0f}, threshold={area_threshold:.0f}")

            if contour_area < area_threshold:
                cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)
            else:
                cv2.drawContours(edgelines, [cnts[i]], 0, (1, 1, 1), -1)
                edgelines = edgelines * edged
                break
            cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)

        # Enhance edge lines for better line detection
        kernel = np.ones((2, 2), np.uint8)
        edgelines = cv2.morphologyEx(edgelines, cv2.MORPH_CLOSE, kernel)
        edgelines = cv2.dilate(edgelines, kernel, iterations=1)

        # Detect lines using Hough transform with multiple thresholds
        lines = None
        thresholds = [150, 120, 100, 80, 60]  # Try progressively lower thresholds

        for threshold in thresholds:
            lines = cv2.HoughLines(edgelines, 1, np.pi / 180, threshold)
            if lines is not None and len(lines) >= 4:
                logger.debug(f"Found {len(lines)} lines with threshold {threshold}")
                break

        if lines is None or len(lines) < 4:
            raise CornerDetectionError(f"Insufficient lines detected: {len(lines) if lines is not None else 0}")

        # Filter and select strong lines with improved logic
        strong_lines = np.zeros([4, 1, 2])
        n2 = 0

        for n1 in range(0, len(lines)):
            if n2 == 4:
                break
            for rho, theta in lines[n1]:
                if n1 == 0:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1
                else:
                    # Check for parallel lines (opposite direction)
                    c1 = np.isclose(abs(rho), abs(strong_lines[0:n2, 0, 0]), atol=100)
                    c2 = np.isclose(np.pi - theta, strong_lines[0:n2, 0, 1], atol=np.pi / 24)
                    c = np.all([c1, c2], axis=0)
                    if any(c):
                        continue

                    # Check for similar lines (same direction)
                    closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=60)
                    closeness_theta = np.isclose(theta, strong_lines[0:n2, 0, 1], atol=np.pi / 24)
                    closeness = np.all([closeness_rho, closeness_theta], axis=0)

                    # Accept line if it's sufficiently different and we need more lines
                    if not any(closeness) and n2 < 4:
                        strong_lines[n2] = lines[n1]
                        n2 = n2 + 1

        # If we still don't have 4 lines, try a more lenient approach
        if n2 < 4:
            logger.warning(f"Only found {n2} strong lines with strict filtering, trying lenient approach")
            strong_lines = np.zeros([4, 1, 2])
            n2 = 0

            for n1 in range(0, min(len(lines), 20)):  # Check first 20 lines
                if n2 == 4:
                    break
                for rho, theta in lines[n1]:
                    if n1 == 0:
                        strong_lines[n2] = lines[n1]
                        n2 = n2 + 1
                    else:
                        # More lenient similarity check
                        closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=80)
                        closeness_theta = np.isclose(theta, strong_lines[0:n2, 0, 1], atol=np.pi / 18)
                        closeness = np.all([closeness_rho, closeness_theta], axis=0)

                        if not any(closeness) and n2 < 4:
                            strong_lines[n2] = lines[n1]
                            n2 = n2 + 1

        # Final fallback: try contour-based corner detection
        if n2 < 4:
            logger.warning("Hough line detection failed, trying contour-based approach")
            return _detect_corners_by_contour(img, edged)

        logger.debug(f"Successfully found {n2} strong lines")

        # Convert lines to line segments
        lines1 = np.zeros((len(strong_lines), 4), dtype=int)
        for i in range(0, len(strong_lines)):
            rho, theta = strong_lines[i][0][0], strong_lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            lines1[i][0] = int(x0 + 1000 * (-b))
            lines1[i][1] = int(y0 + 1000 * (a))
            lines1[i][2] = int(x0 - 1000 * (-b))
            lines1[i][3] = int(y0 - 1000 * (a))

        # Find intersection points
        approx = np.zeros((4, 1, 2), dtype=int)
        index = 0
        combs = list(combinations(lines1, 2))

        for twoLines in combs:
            if index >= 4:
                break
            x1, y1, x2, y2 = twoLines[0]
            x3, y3, x4, y4 = twoLines[1]
            x, y = cross_point([x1, y1, x2, y2], [x3, y3, x4, y4])

            if 0 < x < img.shape[1] and 0 < y < img.shape[0]:
                approx[index] = (int(x), int(y))
                index = index + 1

        if index < 4:
            raise CornerDetectionError(f"Only found {index} valid corner points, need 4")

        # Validate corners
        validate_corners(approx)

        # Scale corners back to original image size
        corners = approx * ratio

        logger.debug(f"Successfully detected {len(corners)} corners")
        return corners

    except CornerDetectionError:
        raise
    except Exception as e:
        logger.error(f"Corner detection failed: {e}")
        raise CornerDetectionError(f"Corner detection failed: {e}")

def set_corner(img, r):
    """
    Set rounded corners on the image.

    Args:
        img: Input image
        r: Corner radius

    Returns:
        Image with rounded corners (BGRA format)
    """
    try:
        b_channel, g_channel, r_channel = cv2.split(img)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        row = img.shape[0]
        col = img.shape[1]

        # Create rounded corners
        for i in range(0, r):
            for j in range(0, r):
                if (r - i) * (r - i) + (r - j) * (r - j) > r * r:
                    alpha_channel[i][j] = 0

        for i in range(0, r):
            for j in range(col - r, col):
                if (r - i) * (r - i) + (r - col + j + 1) * (r - col + j + 1) > r * r:
                    alpha_channel[i][j] = 0

        for i in range(row - r, row):
            for j in range(0, r):
                if (r - row + i + 1) * (r - row + i + 1) + (r - j) * (r - j) > r * r:
                    alpha_channel[i][j] = 0

        for i in range(row - r, row):
            for j in range(col - r, col):
                if (r - row + i + 1) * (r - row + i + 1) + (r - col + j + 1) * (r - col + j + 1) > r * r:
                    alpha_channel[i][j] = 0

        img_bgra = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        return img_bgra

    except Exception as e:
        logger.warning(f"Corner rounding failed: {e}")
        return img

def finetune_result(img, ratio):
    """
    Fine-tune the rectified result image.

    Args:
        img: Rectified image
        ratio: Scaling ratio

    Returns:
        Fine-tuned image

    Raises:
        TransformationError: If fine-tuning fails
    """
    try:
        logger.debug("Starting result fine-tuning...")

        # Crop edges
        offset = int(2 * ratio)
        img = img[offset + 15:img.shape[0] - offset,
              int(offset * 2):img.shape[1] - int(offset * 2), :]

        if img.shape[0] == 0 or img.shape[1] == 0:
            raise TransformationError("Image became empty after cropping")

        # Resize to standard ID card proportions
        if img.shape[0] < img.shape[1]:
            img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 856 * 540)))
            r = int(img.shape[1] / 856 * 31.8)
        else:
            img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 540 * 856)))
            r = int(img.shape[1] / 540 * 31.8)

        # Add rounded corners
        img = set_corner(img, r)

        # Ensure landscape orientation
        if img.shape[0] > img.shape[1]:
            img = cv2.transpose(img)
            img = cv2.flip(img, 0)

        logger.debug("Result fine-tuning completed successfully")
        return img

    except Exception as e:
        logger.error(f"Result fine-tuning failed: {e}")
        raise TransformationError(f"Result fine-tuning failed: {e}")

def rectify_card_image(image, model=None, device=None):
    """
    Main function to rectify a card image using the complete pipeline.

    This is the primary interface function that should be called from the Flask app.
    It handles the entire process from edge detection to final rectification.

    Args:
        image: Input image as numpy array (BGR format)
        model: Trained PyTorch model for CNN edge detection (optional)
        device: PyTorch device (cuda/cpu) (optional)

    Returns:
        Rectified card image as numpy array

    Raises:
        CardRectificationError: If any step of the rectification process fails
    """
    try:
        logger.info("Starting card rectification pipeline...")

        # Validate input
        if image is None:
            raise CardRectificationError("Input image is None")

        if len(image.shape) != 3:
            raise CardRectificationError("Input image must be a 3-channel color image")

        original_height, original_width = image.shape[:2]
        logger.debug(f"Input image size: {original_width}x{original_height}")

        # Resize image for processing
        img_resized = cv2.resize(image, (PROCESS_SIZE, int(PROCESS_SIZE * image.shape[0] / image.shape[1])))
        ratio = image.shape[1] / PROCESS_SIZE

        logger.debug(f"Processing size: {img_resized.shape}, ratio: {ratio:.3f}")

        # Step 1: Edge Detection
        edged = None
        detection_method = "unknown"

        # Try CNN-based edge detection first (if model is available)
        if model is not None and device is not None:
            try:
                logger.debug("Attempting CNN-based edge detection...")
                edged = detect_edge_cnn(image, model, device)
                detection_method = "CNN"
                logger.info("✓ CNN edge detection successful")
            except EdgeDetectionError as e:
                logger.warning(f"CNN edge detection failed: {e}")
                edged = None

        # Fallback to traditional edge detection
        if edged is None:
            try:
                logger.debug("Attempting traditional edge detection...")
                edged = detect_edge_traditional(img_resized)
                detection_method = "traditional"
                logger.info("✓ Traditional edge detection successful")
            except EdgeDetectionError as e:
                logger.error(f"Traditional edge detection failed: {e}")
                raise CardRectificationError("Both CNN and traditional edge detection failed")

        # Step 2: Corner Detection
        try:
            logger.debug("Detecting corners...")
            corners = detect_corners(edged, img_resized, ratio)
            logger.info(f"✓ Corner detection successful using {detection_method} method")
        except CornerDetectionError as e:
            logger.error(f"Corner detection failed: {e}")
            raise CardRectificationError(f"Corner detection failed: {e}")

        # Step 3: Perspective Transformation
        try:
            logger.debug("Performing perspective transformation...")
            result = four_point_transform(image, corners.reshape(4, 2))

            if result is None or result.shape[0] == 0 or result.shape[1] == 0:
                raise TransformationError("Perspective transformation produced empty result")

            logger.debug(f"Transformation result size: {result.shape}")
        except Exception as e:
            logger.error(f"Perspective transformation failed: {e}")
            raise CardRectificationError(f"Perspective transformation failed: {e}")

        # Step 4: Fine-tuning
        try:
            logger.debug("Fine-tuning result...")
            final_result = finetune_result(result, ratio)

            if final_result is None or final_result.shape[0] == 0 or final_result.shape[1] == 0:
                raise TransformationError("Fine-tuning produced empty result")

            logger.info(f"✓ Card rectification completed successfully")
            logger.info(f"Final result size: {final_result.shape}")

            return final_result

        except TransformationError as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise CardRectificationError(f"Fine-tuning failed: {e}")

    except CardRectificationError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in card rectification: {e}")
        raise CardRectificationError(f"Unexpected error: {e}")

def validate_input_image(image):
    """
    Validate input image for card rectification.

    Args:
        image: Input image as numpy array

    Returns:
        True if valid

    Raises:
        CardRectificationError: If image is invalid
    """
    try:
        if image is None:
            raise CardRectificationError("Image is None")

        if not isinstance(image, np.ndarray):
            raise CardRectificationError("Image must be a numpy array")

        if len(image.shape) != 3:
            raise CardRectificationError("Image must be a 3-channel color image")

        height, width, channels = image.shape

        if channels != 3:
            raise CardRectificationError(f"Image must have 3 channels, got {channels}")

        if height < 100 or width < 100:
            raise CardRectificationError(f"Image too small: {width}x{height} (minimum 100x100)")

        if height > 5000 or width > 5000:
            raise CardRectificationError(f"Image too large: {width}x{height} (maximum 5000x5000)")

        # Check if image is not completely black or white
        mean_intensity = np.mean(image)
        if mean_intensity < 5:
            raise CardRectificationError("Image appears to be completely black")

        if mean_intensity > 250:
            raise CardRectificationError("Image appears to be completely white")

        logger.debug(f"Image validation passed: {width}x{height}, mean intensity: {mean_intensity:.1f}")
        return True

    except CardRectificationError:
        raise
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        raise CardRectificationError(f"Image validation failed: {e}")

# Convenience function for backward compatibility
def rectify_card(image, model=None, device=None):
    """
    Convenience wrapper for rectify_card_image.
    Maintains backward compatibility with existing code.
    """
    return rectify_card_image(image, model, device)
