import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d

# Make local imports work like in test.py
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from Pointfilter_Network_Architecture import pointfilternet
from Pointfilter_DataLoader import PointcloudPatchDataset
from Pointfilter_Utils import parse_arguments


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_pointfilter_on_npy(opt, shape_basename):
    """Core of test.py but restricted to a single shape name (without .npy)."""

    # 1) Prepare results dir
    os.makedirs(opt.save_dir, exist_ok=True)

    # 2) Load original noisy points (already saved as .npy)
    original_noise_pts = np.load(os.path.join(opt.testset, shape_basename + ".npy"))
    np.save(
        os.path.join(opt.save_dir, shape_basename + "_pred_iter_0.npy"),
        original_noise_pts.astype("float32"),
    )

    # 3) Iterative filtering like test.py
    for eval_index in range(opt.eval_iter_nums):
        print(f"[INFO] Iteration {eval_index} for {shape_basename}")

        test_dataset = PointcloudPatchDataset(
            root=opt.save_dir,
            shape_name=shape_basename + "_pred_iter_" + str(eval_index),
            patch_radius=opt.patch_radius,
            train_state="evaluation",
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=opt.batchSize, num_workers=int(opt.workers)
        )

        # Load network
        pointfilter_eval = pointfilternet().to(DEVICE)
        model_filename = os.path.join(opt.eval_dir, "model_full_ae.pth")
        checkpoint = torch.load(model_filename, map_location="cpu")
        pointfilter_eval.load_state_dict(checkpoint["state_dict"])
        pointfilter_eval.eval()

        patch_radius = test_dataset.patch_radius_absolute
        pred_pts = np.empty((0, 3), dtype="float32")

        with torch.no_grad():
            for batch_ind, data_tuple in enumerate(test_dataloader):
                noise_patch, noise_inv, noise_disp = data_tuple
                noise_patch = noise_patch.float().to(
                    DEVICE
                )  # (B, 3, K) after transpose
                noise_inv = noise_inv.float().to(DEVICE)
                noise_patch = noise_patch.transpose(
                    2, 1
                ).contiguous()  # (B, K, 3) -> (B, 3, K)

                predict = pointfilter_eval(noise_patch)  # (B, 3)
                predict = predict.unsqueeze(2)  # (B, 3, 1)
                predict = torch.bmm(noise_inv, predict)  # (B, 3, 1)

                pts_batch = (
                    np.squeeze(predict.cpu().numpy()) * patch_radius
                    + noise_disp.numpy()
                )
                pred_pts = np.append(pred_pts, pts_batch, axis=0)

        np.save(
            os.path.join(
                opt.save_dir,
                shape_basename + "_pred_iter_" + str(eval_index + 1) + ".npy",
            ),
            pred_pts.astype("float32"),
        )

    # Return final iteration .npy path
    return os.path.join(
        opt.save_dir, shape_basename + f"_pred_iter_{opt.eval_iter_nums}.npy"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Pointfilter on a single .ply point cloud."
    )
    parser.add_argument("--input", required=True, help="Input .ply path.")
    parser.add_argument("--output", required=True, help="Output .ply path.")
    parser.add_argument(
        "--eval_dir",
        default="./Summary/pre_train_model",
        help="Directory with model_full_ae.pth.",
    )
    parser.add_argument(
        "--tmp_root",
        default="./Dataset/CustomSingle",
        help="Temporary root to store .npy and intermediate results.",
    )
    args = parser.parse_args()

    in_ply = args.input
    out_ply = args.output

    if not os.path.isfile(in_ply):
        raise FileNotFoundError(f"Input file not found: {in_ply}")

    # 1) Load input .ply
    print(f"[INFO] Loading input PLY: {in_ply}")
    pcd = o3d.io.read_point_cloud(in_ply)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Input point cloud has no points.")
    print(f"[INFO] Points: {pts.shape[0]}")

    # 2) Create temp dirs like test.py expects
    testset_dir = os.path.join(args.tmp_root, "Test")
    save_dir = os.path.join(args.tmp_root, "Results")
    os.makedirs(testset_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 3) Choose a shape name from the input filename (without extension)
    shape_basename = os.path.splitext(os.path.basename(in_ply))[0]

    # Save noisy points as <name>.npy in testset
    np.save(os.path.join(testset_dir, shape_basename + ".npy"), pts.astype("float32"))

    # 4) Build a minimal opt/parameters object based on test.py defaults
    from types import SimpleNamespace

    opt = SimpleNamespace()
    opt.testset = testset_dir
    opt.eval_dir = args.eval_dir
    opt.batchSize = 64
    opt.workers = 8
    opt.save_dir = save_dir
    opt.eval_iter_nums = 2
    opt.patch_radius = 0.05

    print(f"[INFO] Running Pointfilter pipeline on {shape_basename}...")
    final_npy = run_pointfilter_on_npy(opt, shape_basename)

    # 5) Load final filtered points and write PLY
    filtered_pts = np.load(final_npy)
    print(f"[INFO] Final filtered points: {filtered_pts.shape[0]}")

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(filtered_pts)

    # Optionally copy colors if present and same length
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if len(colors) == filtered_pts.shape[0]:
            pcd_out.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pcd_out)
    print(f"[INFO] Saved cleaned PLY to: {out_ply}")


if __name__ == "__main__":
    main()
