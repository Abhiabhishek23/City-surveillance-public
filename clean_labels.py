import os

dataset_path = "/Users/abhishekx/Desktop/City-surveillance/Data/crowd/dataset/labels/train"
valid_classes = [0]  # Only 1 class for crowd

for file in os.listdir(dataset_path):
    if file.endswith(".txt"):
        path = os.path.join(dataset_path, file)
        lines = []
        with open(path, "r") as f:
            for line in f.readlines():
                cls = int(line.split()[0])
                if cls in valid_classes:
                    lines.append(line)
        if lines:
            with open(path, "w") as f:
                f.writelines(lines)
        else:
            os.remove(path)  # Remove label file if no valid classes
            # Optionally remove corresponding image
            img_file = os.path.join(dataset_path.replace("labels", "images"), file.replace(".txt", ".jpg"))
            if os.path.exists(img_file):
                os.remove(img_file)
