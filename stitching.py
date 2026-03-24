'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
from numpy import degrees
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #

# helper functions
def to_float_bchw(img: torch.Tensor) -> torch.Tensor:
    '''
    Convert image to float and add batch dimension.
    Args:
        img: CxHxW image tensor
    Returns:
        img: 1xCxHxW image tensor
    '''
    if img.dtype != torch.float32:
        img = img.float() / 255.0
    return img.unsqueeze(0)

def to_grayscale(img_bchw: torch.Tensor) -> torch.Tensor:
    '''
    Convert RGB image to grayscale for feature extraction.
    Args:
        img_bchw: RGB image in 1xCxHxW format
    Returns:
        gray_bchw: grayscale image in 1x1xHxW format
    '''
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


def blend_two_images(img1_warped, img2_warped, mask1, mask2):
    """
    Blend two warped images together using their masks to handle overlaps.
    Args:
        img1_warped: warped image 1
        img2_warped: warped image 2
        mask1: mask for image 1
        mask2: mask for image 2
    Returns:
        blended_image: the blended image
    """
    overlap = mask1 * mask2

    # combining if no overlap
    if overlap.sum() == 0:
        return img1_warped * mask1 + img2_warped * mask2

    # finding overlap region
    coords = torch.nonzero(overlap[0, 0], as_tuple=False)

    x_min = coords[:, 1].min().item()
    x_max = coords[:, 1].max().item()

    seam_x = x_min + 0.6 * (x_max - x_min)

    # building mask
    _, _, H, W = img1_warped.shape
    x_coords = torch.arange(W, device=img1_warped.device).view(1, 1, 1, W)

    mask_left = (x_coords <= seam_x).float()
    mask_right = 1.0 - mask_left

    # applying masks
    img1_part = img1_warped * mask1 * mask_left
    img2_part = img2_warped * mask2 * mask_right

    return img1_part + img2_part

def compute_pairwise_homography(lafs1, desc1, lafs2, desc2, min_matches: int = 20, min_inliers: int = 15):
    """
    Compute the homography between two images given their local affine frames and descriptors.
    Args:
        lafs1: local affine frames for image 1
        desc1: descriptors for image 1
        lafs2: local affine frames for image 2
        desc2: descriptors for image 2
        min_matches: minimum number of matches required
        min_inliers: minimum number of inliers required
    Returns:
        H_21: 1x3x3 homography or None
        is_overlap: bool
        num_inliers: int
    """
    distances, idxs = match_descriptors(desc1, desc2)

    if idxs.shape[0] < min_matches:
        return None, False, 0

    pts1_all = lafs_to_center_points(lafs1)
    pts2_all = lafs_to_center_points(lafs2)

    matched_pts1 = pts1_all[idxs[:, 0]]
    matched_pts2 = pts2_all[idxs[:, 1]]

    H_21, inliers = compute_homography_ransac(matched_pts1, matched_pts2)

    num_inliers = int(inliers.sum().item())
    if num_inliers < min_inliers:
        return None, False, num_inliers

    matched_pts1 = matched_pts1[inliers]
    matched_pts2 = matched_pts2[inliers]

    H_21 = K.geometry.find_homography_dlt(
        matched_pts2.unsqueeze(0),
        matched_pts1.unsqueeze(0)
    )

    return H_21, True, num_inliers

def invert_homography(H: torch.Tensor) -> torch.Tensor:
    """
    Invert a homography matrix.
    Args:
        H: 1x3x3 homography matrix
    Returns:
        H_inv: 1x3x3 inverted homography matrix
    """
    if H.dim() == 3:
        H = H[0]
    H_inv = torch.linalg.inv(H)
    return H_inv.unsqueeze(0)

def build_global_canvas(images_bchw, transforms_to_ref):
    """
    Compute the size of the output canvas needed to fit all images after warping, and the translation homography to shift everything into positive coordinates.
    Args:
        images_bchw: list of images in BCHW format
        transforms_to_ref: list of homographies mapping each image to the reference frame
    Returns:
        out_h: height of the output canvas
        out_w: width of the output canvas
        T: translation homography
    """
    device = images_bchw[0].device
    dtype = images_bchw[0].dtype

    all_corners = []

    for img_bchw, H in zip(images_bchw, transforms_to_ref):
        if H is None:
            continue
        _, _, h, w = img_bchw.shape
        corners = warp_corners(h, w, H, device, dtype)
        all_corners.append(corners)

    all_corners = torch.cat(all_corners, dim=0)

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

def blend_multi_image(warped_images, warped_masks) -> torch.Tensor:
    """
    Blend multiple warped images together using their masks to handle overlaps.
    Args:
        warped_images: list of warped images
        warped_masks: list of masks for the warped images
    Returns:
        blended: the blended image
    """
    imgs = torch.cat(warped_images, dim=0)
    masks = torch.cat(warped_masks, dim=0)

    K, _, H, W = imgs.shape

    # coordinate grid
    y, x = torch.meshgrid(
        torch.arange(H, device=imgs.device),
        torch.arange(W, device=imgs.device),
        indexing="ij"
    )
    y = y.float()
    x = x.float()

    # approximating center of each warped image
    centers = []
    for i in range(K):
        mask = masks[i, 0]
        ys, xs = torch.where(mask > 0.5)
        if ys.numel() == 0:
            centers.append((H / 2.0, W / 2.0))
        else:
            centers.append((ys.float().mean(), xs.float().mean()))

    # distance to center score for each image
    dist_stack = []
    for i in range(K):
        cy, cx = centers[i]
        dist = (y - cy) ** 2 + (x - cx) ** 2
        dist_stack.append(dist.unsqueeze(0))
    dist_stack = torch.cat(dist_stack, dim=0)

    # ignoring invalid pixels by setting them to a large distance
    dist_stack = dist_stack + (1.0 - masks[:, 0]) * 1e10

    # tracking best and second best sources per pixel
    sorted_dist, sorted_idx = torch.sort(dist_stack, dim=0)
    best_idx = sorted_idx[0]
    second_idx = sorted_idx[1]

    # gathering best image
    best_idx_rgb = best_idx.unsqueeze(0).unsqueeze(0).expand(1, 3, H, W)
    best_img = torch.gather(imgs, dim=0, index=best_idx_rgb).squeeze(0)

    # gathering second best image
    second_idx_rgb = second_idx.unsqueeze(0).unsqueeze(0).expand(1, 3, H, W)
    second_img = torch.gather(imgs, dim=0, index=second_idx_rgb).squeeze(0)

    # calculating confidence for seam
    margin = sorted_dist[1] - sorted_dist[0]

    # applying small margin band around seams
    margin_width = 4500.0
    alpha = torch.clamp(margin / margin_width, 0.0, 1.0)
    alpha = alpha.unsqueeze(0)

    # mixing best and second best
    blended = best_img * alpha + second_img * (1.0 - alpha)

    # zeroing out pixels where nothing is valid
    valid = (masks.sum(dim=0) > 0).float()
    blended = blended * valid.expand_as(blended)

    return blended.unsqueeze(0)

def normalize_homography(H: torch.Tensor) -> torch.Tensor:
    """
    Normalize a homography matrix so that the bottom-right value is 1.
    Args:
        H: 1x3x3 or 3x3 homography matrix
    Returns:
        H_normalized: normalized homography matrix
    """
    if H.dim() == 3:
        H = H[0]
    H = H / H[2, 2].clamp(min=1e-8)
    return H.unsqueeze(0)

def make_affine(H: torch.Tensor) -> torch.Tensor:
    """
    Convert a homography to an affine transformation by zeroing out the projective components.
    Args:
        H: 1x3x3 or 3x3 homography matrix
    Returns:
        H_affine: 1x3x3 affine transformation matrix
    """
    if H.dim() == 3:
        H = H[0]

    H[2, 0] = 0.0
    H[2, 1] = 0.0
    H[2, 2] = 1.0

    return H.unsqueeze(0)

# main task 1 function
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: torch.Tensor of the stitched background image
    """
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

    # using RANSAC to find inliers
    H_21, inliers = compute_homography_ransac(matched_pts1, matched_pts2)

    if inliers.sum() < 4:
        return img1

    # keep only inlier correspondences once
    matched_pts1 = matched_pts1[inliers]
    matched_pts2 = matched_pts2[inliers]

    # recompute homography using only inliers
    H_21 = K.geometry.find_homography_dlt(
        matched_pts2.unsqueeze(0),
        matched_pts1.unsqueeze(0)
    )

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
    img = blend_two_images(img1_warped, img2_warped, mask1_warped, mask2_warped)

    # converting back to uint8 format and removing batch dimension
    img = img.squeeze(0).clamp(0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)

    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    # img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    # overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    names = sorted(list(imgs.keys()))
    n = len(names)

    if n == 0:
        img = torch.zeros((3, 256, 256), dtype=torch.uint8)
        overlap = torch.zeros((0, 0), dtype=torch.int64)
        return img, overlap

    # Initializing images, grayscale, features
    images_bchw = []
    lafs_list = []
    desc_list = []

    for name in names:
        img = imgs[name]
        img_bchw = to_float_bchw(img)
        gray = to_grayscale(img_bchw)
        lafs, descs = extract_sift_features(gray, num_features=1000)

        images_bchw.append(img_bchw)
        lafs_list.append(lafs)
        desc_list.append(descs[0])  # remove batch dim

    # overlapping matrix and homographies pairwise
    overlap = torch.eye(n, dtype=torch.int64)
    pair_H = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            H_ji, is_overlap, num_inliers = compute_pairwise_homography(
                lafs_list[i], desc_list[i],
                lafs_list[j], desc_list[j],
                min_matches=20,
                min_inliers=15
            )

            if is_overlap:
                overlap[i, j] = 1
                overlap[j, i] = 1

                pair_H[i][j] = H_ji
                pair_H[j][i] = invert_homography(H_ji)

    # determining reference frame
    ref_degree = overlap.sum(dim=1)
    ref_idx = int(torch.argmax(ref_degree))

    # computing homographies to the reference frame using BFS
    transforms_to_ref = [None for _ in range(n)]
    transforms_to_ref[ref_idx] = torch.eye(3, dtype=images_bchw[0].dtype, device=images_bchw[0].device).unsqueeze(0)

    visited = [False for _ in range(n)]
    queue = [ref_idx]
    visited[ref_idx] = True

    while len(queue) > 0:
        cur = queue.pop(0)

        for nxt in range(n):
            if pair_H[cur][nxt] is None:
                continue
            if visited[nxt]:
                continue

            if pair_H[ref_idx][nxt] is not None:
                transforms_to_ref[nxt] = normalize_homography(pair_H[ref_idx][nxt])
            else:
                H_new = transforms_to_ref[cur] @ pair_H[cur][nxt]
                transforms_to_ref[nxt] = normalize_homography(H_new)

            visited[nxt] = True
            queue.append(nxt)

    # keeping only connected images
    connected_indices = [i for i in range(n) if transforms_to_ref[i] is not None]

    if len(connected_indices) == 0:
        img = imgs[names[0]]
        return img, overlap
    
    valid = torch.zeros_like(overlap)
    for i in connected_indices:
        for j in connected_indices:
            valid[i, j] = overlap[i, j]
    overlap = valid

    # building output canvas from connected images only
    connected_images = [images_bchw[i] for i in connected_indices]
    connected_transforms = [transforms_to_ref[i] for i in connected_indices]

    out_h, out_w, T = build_global_canvas(connected_images, connected_transforms)

    warped_images = []
    warped_masks = []

    for i in connected_indices:
        img_bchw = images_bchw[i]
        H = T @ transforms_to_ref[i]

        warped = K.geometry.transform.warp_perspective(
            img_bchw, H, dsize=(out_h, out_w),
            mode="bilinear", padding_mode="zeros", align_corners=True
        )

        mask = torch.ones(
            (1, 1, img_bchw.shape[-2], img_bchw.shape[-1]),
            dtype=img_bchw.dtype,
            device=img_bchw.device
        )

        warped_mask = K.geometry.transform.warp_perspective(
            mask, H, dsize=(out_h, out_w),
            mode="nearest", padding_mode="zeros", align_corners=True
        )

        warped_mask = (warped_mask > 0.5).float()

        warped_images.append(warped)
        warped_masks.append(warped_mask)

    img = blend_multi_image(warped_images, warped_masks)

    img = img.squeeze(0).clamp(0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)

    return img, overlap
