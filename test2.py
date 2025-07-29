import os

# Full expected path to the shared folder
shared_folder = r"C:\Users\ly.hoangminhdatdylan\OneDrive - Spartronics\NguyenNgoc, Lam's files - Dylan Project"

if os.path.exists(shared_folder) and os.path.isdir(shared_folder):
    print("✅ Shared folder exists and is accessible.")
else:
    print("❌ Folder does not exist or is not synced yet.")
