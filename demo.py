import glob

train_files = glob.glob("../susant/Knee_Multicoil_train_batch0/**/*.h5", recursive=True)
val_files = glob.glob("../susant/Knee_Multicoil_train_batch1/**/*.h5", recursive=True)

print("Training volumes:", len(train_files))
print("Validation volumes:", len(val_files))
