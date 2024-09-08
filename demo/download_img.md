### Function to Download image from url and save it to disk

```py
import os
from PIL import Image

def download_image_with_curl(url, save_path):
    try:
        # Using curl to download the image
        os.system(f"curl -o {save_path} {url}")
        # Open the image using PIL
        image = Image.open(save_path)
        image.verify()  # Verify that it is a valid image
        return image
    except Exception as e:
        print(f"Error downloading or processing the image: {e}")
        return None

# URLs for the images
path = '/content'
urls = {
    'source_image': 'https://filmfare.wwmindia.com/content/2020/jul/shahrukhkhan41596116759.jpg',  # Replace with actual URLs
    'target_image': 'https://cdn.siasat.com/wp-content/uploads/2024/01/Sallu.jpg',
    'family_photo': 'https://feeds.abplive.com/onecms/images/uploaded-images/2023/04/18/19a3085bf722da7c75f9708da30bc8d71681789348745431_original.jpg?impolicy=abp_cdn&imwidth=1200&height=675',
}

# Paths to save images locally
path = '/content'
image_paths = {
    "source_image": f"{path}/source_image.jpg",
    "target_image": f"{path}/target_image.jpg",
    "family_photo": f"{path}/family_photo.jpg"
}

# Download and process images
source_image = download_image_with_curl(urls['source_image'], image_paths['source_image'])
target_image = download_image_with_curl(urls['target_image'], image_paths['target_image'])
family_photo = download_image_with_curl(urls['family_photo'], image_paths['family_photo'])

# Check if the family photo was successfully downloaded and processed
if family_photo is None:
    print("Failed to download or identify the family photo.")
else:
    print(f"Family photo successfully downloaded and verified.")
```

### Display the images

```py
from IPython.display import Image, display
# Display the images
display(Image(filename=image_paths['source_image']))
display(Image(filename=image_paths['target_image']))
display(Image(filename=image_paths['family_photo']))
```

### Display the images in one row

```py
import matplotlib.pyplot as plt

# Display the images in one row
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(plt.imread(image_paths['source_image']))
axs[0].axis('off')
axs[1].imshow(plt.imread(image_paths['target_image']))
axs[1].axis('off')
axs[2].imshow(plt.imread(image_paths['family_photo']))
axs[2].axis('off')
plt.show()
```
