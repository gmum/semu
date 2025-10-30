import os

path = "../DDPM/results/cifar10/2024_12_30_054526/fid_samples_guidance_2.0/salun2/"

for i in range(15000):
    os.rename(f"{path}{i}.png", f"{path}{i+30000}.png")