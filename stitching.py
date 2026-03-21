'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #

# helper functions for task 1
def to_float_bchw(img: torch.Tensor) -> torch.Tensor:
    # Convert image to float and add batch dimension
    if img.dtype != torch.float32:
        img = img.float() / 255.0
    return img.unsqueeze(0)

def to_grayscale(img_bchw: torch.Tensor) -> torch.Tensor:
    # Convert RGB image to grayscale for feature extraction
    return K.color.rgb_to_grayscale(img_bchw)

def extract_sift_features(gray_bchw: torch.Tensor, num_features: int = 800):
    """
    Extract SIFT features from a grayscale image.

    Args:
        gray_bchw: grayscale image in 1x1xHxW format
        num_features: number of features to extract 
    Returns:
        lafs: local affine frames for the keypoints
        descs: SIFT descriptors for the keypoints 
    """
    # SIFTFeature does detection + description
    sift = K.feature.SIFTFeature(num_features=num_features)
    lafs, responses, descs = sift(gray_bchw)
    return lafs, descs

def match_descriptors(desc1: torch.Tensor, desc2: torch.Tensor):
    """
    Match descriptors between two sets using SuperPoint Mutual Nearest Neighbor (SMNN) matching.
    Args:
        desc1: descriptors from image 1
        desc2: descriptors from image 2
    Returns:
        distances: matching distances for each match
        idxs: indices of matched descriptors in desc2 for each descriptor in desc1
    """
    distances, idxs = K.feature.match_smnn(desc1, desc2, th=0.95)
    return distances, idxs

def lafs_to_center_points(lafs: torch.Tensor) -> torch.Tensor:
    """
    Convert local affine frames to center points.
    Args:
        lafs: local affine frames in Nx2x3 format
    Returns:
        centers: Nx2 tensor of keypoint center coordinates
    """
    # The translation column contains the keypoint centers
    return lafs[0, :, :, 2]

def compute_homography_ransac(kpts1: torch.Tensor, kpts2: torch.Tensor):
    """
    Compute homography using RANSAC given matched keypoints.
    Args:
        kpts1: Nx2 tensor of matched keypoints from image 1
        kpts2: Nx2 tensor of matched keypoints from image 2
    Returns:
        H_21: 1x3x3 homography matrix mapping points from image 2 to image 1
        inlier_mask: boolean tensor indicating which matches are inliers
    """
    ransac = K.geometry.ransac.RANSAC(
        model_type="homography",
        inl_th=3.0,
        batch_size=2048,
        max_iter=2000,
        confidence=0.999,
        max_lo_iters=10,
    )

    H_21, inlier_mask = ransac(kpts2, kpts1)

    if H_21.dim() == 2:
        H_21 = H_21.unsqueeze(0)

    return H_21, inlier_mask

def warp_corners(h: int, w: int, H: torch.Tensor, device, dtype):
    """
    Warp the corners of the image using the homography H.
    Args:
        h: height of the image
        w: width of the image
        H: 1x3x3 or 3x3 homography matrix
        device: device to create the corners tensor on
        dtype: data type for the corners tensor
    Returns:
        warped_corners: 4x2 tensor of the warped corner coordinates
    """
    corners = torch.tensor(
        [[0.0, 0.0],
         [w - 1.0, 0.0],
         [w - 1.0, h - 1.0],
         [0.0, h - 1.0]],
        device=device,
        dtype=dtype
    )

    if H.dim() == 3:
        H = H[0]

    ones = torch.ones((corners.shape[0], 1), device=device, dtype=dtype)
    corners_h = torch.cat([corners, ones], dim=1)

    warped_h = (H @ corners_h.t()).t()
    warped_xy = warped_h[:, :2] / warped_h[:, 2:3].clamp(min=1e-8)

    return warped_xy

def build_output_canvas(img1_bchw: torch.Tensor, img2_bchw: torch.Tensor, H_21: torch.Tensor):
    """
    Compute the size of the output canvas needed to fit both images after warping, and the translation homography to shift everything into positive coordinates.
    Args:
        img1_bchw: image 1 in BCHW format
        img2_bchw: image 2 in BCHW format
        H_21: homography mapping image 2 to image 1
    Returns:
        out_h: height of the output canvas
        out_w: width of the output canvas
        T: translation homography
    """
    _, _, h1, w1 = img1_bchw.shape
    _, _, h2, w2 = img2_bchw.shape
    device = img1_bchw.device
    dtype = img1_bchw.dtype

    c1 = torch.tensor(
        [[0.0, 0.0],
         [w1 - 1.0, 0.0],
         [w1 - 1.0, h1 - 1.0],
         [0.0, h1 - 1.0]],
        device=device,
        dtype=dtype
    )

    c2w = warp_corners(h2, w2, H_21, device, dtype)

    all_corners = torch.cat([c1, c2w], dim=0)
    min_xy = torch.floor(all_corners.min(dim=0).values)
    max_xy = torch.ceil(all_corners.max(dim=0).values)

    min_x, min_y = min_xy[0], min_xy[1]
    max_x, max_y = max_xy[0], max_xy[1]

    tx = -min_x
    ty = -min_y

    T = torch.tensor(
        [[[1.0, 0.0, tx],
          [0.0, 1.0, ty],
          [0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype
    )

    out_w = int((max_x - min_x + 1).item())
    out_h = int((max_y - min_y + 1).item())

    return out_h, out_w, T


def blend_two_images(img1_warped: torch.Tensor, img2_warped: torch.Tensor,
                      mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """
    Blend two warped images together using their masks to handle overlaps.
    Args:
        img1_warped: warped image 1 in the output canvas frame
        img2_warped: warped image 2 in the output canvas frame
        mask1: binary mask indicating valid pixels in img1_warped
        mask2: binary mask indicating valid pixels in img2_warped
    Returns:
        blended: blended output image
    """
    weight = mask1 + mask2
    weight = torch.clamp(weight, min=1.0)
    blended = (img1_warped * mask1 + img2_warped * mask2) / weight
    return blended

# main task 1 function
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    names = sorted(list(imgs.keys()))
    assert len(names) == 2, "Task 1 expects exactly two input images."

    img1 = imgs[names[0]]
    img2 = imgs[names[1]]

    # converting to float and add batch dimension
    img1_bchw = to_float_bchw(img1)
    img2_bchw = to_float_bchw(img2)

    # converting to grayscale for feature extraction
    gray1 = to_grayscale(img1_bchw)
    gray2 = to_grayscale(img2_bchw)

    # detecting and describing features with SIFT
    lafs1, descs1 = extract_sift_features(gray1, num_features=800)
    lafs2, descs2 = extract_sift_features(gray2, num_features=800)

    # removing batch dimension to go from 1xNxd to Nxd
    desc1 = descs1[0]
    desc2 = descs2[0]

    # matching descriptors between the two images
    distances, idxs = match_descriptors(desc1, desc2)

    if idxs.shape[0] < 4:
        # if not enough matches to estimate homography
        return img1

    # converting matched local affine frames to center points for homography estimation
    pts1_all = lafs_to_center_points(lafs1)  # Nx2
    pts2_all = lafs_to_center_points(lafs2)  # Nx2

    matched_pts1 = pts1_all[idxs[:, 0]]
    matched_pts2 = pts2_all[idxs[:, 1]]

    # estimating homography using RANSAC to find inliers
    H_21, inliers = compute_homography_ransac(matched_pts1, matched_pts2)

    if inliers.sum() < 4:
        return img1

    matched_pts1 = matched_pts1[inliers]
    matched_pts2 = matched_pts2[inliers]

    # computing the size of the output canvas and the translation homography
    out_h, out_w, T = build_output_canvas(img1_bchw, img2_bchw, H_21)

    # computing the size of the output canvas and the translation homography
    out_h, out_w, T = build_output_canvas(img1_bchw, img2_bchw, H_21)

    # identity homography for image 1
    H1 = T
    # warping image2 into image1 frame and then translating into canvas coordinates
    H2 = T @ H_21

    # warping both images into the output canvas frame
    img1_warped = K.geometry.transform.warp_perspective(
        img1_bchw, H1, dsize=(out_h, out_w),
        mode="bilinear", padding_mode="zeros", align_corners=True
    )

    img2_warped = K.geometry.transform.warp_perspective(
        img2_bchw, H2, dsize=(out_h, out_w),
        mode="bilinear", padding_mode="zeros", align_corners=True
    )

    # warping masks to identify valid pixels in the warped images
    mask1 = torch.ones((1, 1, img1_bchw.shape[-2], img1_bchw.shape[-1]),
                       dtype=img1_bchw.dtype, device=img1_bchw.device)
    mask2 = torch.ones((1, 1, img2_bchw.shape[-2], img2_bchw.shape[-1]),
                       dtype=img2_bchw.dtype, device=img2_bchw.device)

    mask1_warped = K.geometry.transform.warp_perspective(
        mask1, H1, dsize=(out_h, out_w),
        mode="nearest", padding_mode="zeros", align_corners=True
    )

    mask2_warped = K.geometry.transform.warp_perspective(
        mask2, H2, dsize=(out_h, out_w),
        mode="nearest", padding_mode="zeros", align_corners=True
    )

    mask1_warped = (mask1_warped > 0.5).float()
    mask2_warped = (mask2_warped > 0.5).float()

    # blending the two warped images together using their masks to handle overlaps
    stitched = blend_two_images(img1_warped, img2_warped, mask1_warped, mask2_warped)

    # converting back to uint8 format and removing batch dimension
    stitched = stitched.squeeze(0).clamp(0.0, 1.0)
    stitched = (stitched * 255.0).round().to(torch.uint8)

    return stitched

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap
