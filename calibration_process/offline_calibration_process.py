import cv2
from typing import List
import numpy as np
import cv2
import copy
import glob 
import os
import json

# Auxiliar functions
def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]

def write_image(output_folder: str, img_name: str, img: np.array):
    img_path = os.path.join(output_folder, img_name)  
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(img_path, img)  

def show_image(img: np.array, img_name: str = "Image"):
    cv2.imshow(img_name, img)  
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

imgs_gen_path = 'calibration_images/*.jpeg' 
imgs_path = sorted(glob.glob(imgs_gen_path))
imgs = load_images(imgs_path) 

# print(imgs_path)

# Number of rows and cols
pattern_size = (11,8)

corners = [cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), pattern_size) for img in imgs]  

# Security copy
corners_copy = copy.deepcopy(corners)  

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01) 

# To refine corner detections we input grayscale images
imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

corners_refined = [cv2.cornerSubPix(i, cor[1], pattern_size, (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

imgs_copy = copy.deepcopy(imgs) 

output_folder_refined_corners = "refined_corners_output"
for i in range(len(corners_copy)):
    ret = corners_copy[i][0]
    if ret: 
        cv2.drawChessboardCorners(imgs_copy[i], pattern_size, corners_refined[i], ret) 
        write_image(output_folder=output_folder_refined_corners, img_name=f"refined_corners_{i}.jpg", img=imgs_copy[i])


def get_chessboard_points(chessboard_shape, dx, dy):

    cols, rows = chessboard_shape 
    grid_xy = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
    objp = np.zeros([grid_xy.shape[0], 3], dtype=np.float32) 
    objp[:,0] = grid_xy[:,0]*dx  
    objp[:,1] = grid_xy[:,1]*dy  

    return objp  

# Needs to be changed to the real position of our chessboard corners
dx, dy = 16, 16 
chessboard_points = [get_chessboard_points(pattern_size, dx, dy) for _ in corners_refined]  

# Filter data and get only those with adequate detections
valid_corners = [cor[1] for cor in corners if cor[0]]  

# Convert list to numpy array
valid_corners = np.asarray(valid_corners, dtype=np.float32)

# We adjust the number of chessboard_points to be equal to the number of valid corners
chessboard_points = [chessboard_points[i] for i in range(len(valid_corners))]  

img_size = (imgs_gray[0].shape[1], imgs_gray[0].shape[0]) 

rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, valid_corners, img_size, None, None)

# Obtain extrinsics
extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs)) 

# Print outputs
print("Intrinsics:\n", intrinsics)
print("\nDistortion coefficients:\n", dist_coeffs)
print("\nRoot mean squared reprojection error:\n", rms)
print("\nExtrinsics:\n", extrinsics)

def save_calibration_data(mtx, dist, rvecs, tvecs, filename="calibration_data.json"):
    """
    Saves the intrinsic matrix, distortion coefficients, and extrinsic parameters
    (rvecs, tvecs) to a JSON file.
    """
    print("Saving calibration data...")
    
    # Convert NumPy arrays to standard Python lists
    if isinstance(mtx, np.ndarray):
        mtx_list = mtx.tolist()
    else:
        mtx_list = mtx

    if isinstance(dist, np.ndarray):
        # Flatten the distortion coefficients array before converting
        dist_list = np.ravel(dist).tolist() 
    else:
        dist_list = dist

    # Process Extrinsics (rvecs and tvecs are lists of arrays)
    # Convert each array within the list to a Python list
    rvecs_list = []
    if rvecs is not None:
        for r in rvecs:
            if isinstance(r, np.ndarray):
                rvecs_list.append(r.tolist())
            else:
                rvecs_list.append(r)

    tvecs_list = []
    if tvecs is not None:
        for t in tvecs:
            if isinstance(t, np.ndarray):
                tvecs_list.append(t.tolist())
            else:
                tvecs_list.append(t)

    # Create the data dictionary
    data = {
        "intrinsics": mtx_list,
        "distortion": dist_list,
        "extrinsics": {
            "rvecs": rvecs_list,
            "tvecs": tvecs_list
        }
    }

    # Write to file
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Success! Data saved to '{filename}'")
        print(f"   Matrix: \n{np.array(mtx_list)}")
        print(f"   Distortion: {dist_list}")
        print(f"   Extrinsics saved: {len(rvecs_list)} vector pairs.")
    except Exception as e:
        print(f"Error saving the file: {e}")

if 'intrinsics' in locals() and 'dist_coeffs' in locals() and 'rvecs' in locals() and 'tvecs' in locals():
    save_calibration_data(intrinsics, dist_coeffs, rvecs, tvecs)
else:
    print("Saving error: Required calibration variables not found in local scope.")


