# https://huggingface.co/docs/diffusers/conceptual/evaluation
import argparse

import torch
import os
from dataset import setup_fid_data, setup_fid_data_all_at_once, setup_fid_data_all_at_once2
from torchmetrics.image.fid import FrechetInceptionDistance as FID


def compute_fid(class_to_forget, path, label, image_size):
    print(f"LABEL={label}")
    fid = FID(feature=64).cuda()
    real_set, fake_set = setup_fid_data_all_at_once2(class_to_forget, path, label, image_size)
    real_images = torch.stack(real_set).to(torch.uint8).cuda()
    fake_images = torch.stack(fake_set).to(torch.uint8).cuda()

    fid.update(real_images, real=True)  # doctest: +SKIP
    fid.update(fake_images, real=False)  # doctest: +SKIP
    fid_results = fid.compute()
    print(fid_results)
    return fid_results


def compute_fids_in_directory(directory):
        
        folder = directory

        final_images_path = os.path.join(folder, "final_images")   
        common_path_results = os.path.join(folder, "fid_results.txt")

        subdirectories = [ f.path for f in os.scandir(final_images_path) if f.is_dir() ]
        for subdir in subdirectories:
            subdir_last_part = subdir.split("/")[-1]

            my_current_dir = os.path.join(subdir, subdir_last_part)
            count = 0
            for path in os.scandir(my_current_dir):
                if path.is_file() and path.name.endswith(".png"):
                    count += 1
            if count != 1000:
                continue
            
            class_to_forget = list(subdir_last_part.split("class_")[-1])[0]
            class_to_forget = int(class_to_forget)
            print(f"DIR={my_current_dir}")
            print(f"CLASS={class_to_forget}")
            save_path_results = f"{folder}/{subdir_last_part}_results_fid.txt"

            fid_res = compute_fid(class_to_forget, my_current_dir, 512)
            with open(save_path_results, "w+") as f:
                f.write(str(fid_res))
            with open(common_path_results, "a+") as f:
                f.write(f"{subdir_last_part}={fid_res}\n")
        print("Done!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--folder_path", help="path of images", type=str, required=True)
    parser.add_argument(
        "--class_to_forget", help="class_to_forget", type=int, required=False, default=6
    )
    parser.add_argument(
        "--label", help="class_to_forget", type=int, required=False, default=0
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    args = parser.parse_args()

    compute_fid(args.class_to_forget, args.folder_path, args.label, args.image_size)

    # compute_fids_in_directory(args.folder_path)
    # path = args.folder_path
    # class_to_forget = args.class_to_forget
    # image_size = args.image_size
    # print(class_to_forget)
    # compute_fid(class_to_forget, path, image_size)
