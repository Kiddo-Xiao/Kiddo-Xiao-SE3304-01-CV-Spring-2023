import os

labels = os.listdir(r'./labels')
images = os.listdir(r'./images')

to_remove = []
for image in images:
    if image not in labels:
        to_remove.append(image)

print(len(to_remove))

for image in to_remove:
    os.remove(os.path.join(r'./images', image))
    