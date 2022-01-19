# %%
import matplotlib.pyplot as plt
import pathlib
import random
import numpy as np
data_dir = '/storage_data/zhou_Ningkun/cryocheck/train_data'
def show_manual(num_to_display):
    img_paths = list(pathlib.Path(data_dir).glob('*/*/*.png'))
    random.shuffle(img_paths)
    batch = img_paths[:num_to_display]
    width =  round(np.sqrt(num_to_display))
    plt.figure(figsize=(10,10))
    for i in range(1, width*width +1):
        img = plt.imread(batch[i-1])
        title = batch[i-1].parent.name
        plt.subplot(width, width, i)
        plt.imshow(img)
        plt.axis('off')
        if title == 'bad':
            plt.title(title, color='red')
        else:
            plt.title(title, color='green')
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    show_manual(9)

# %%
