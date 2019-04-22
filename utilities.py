import numpy as np
import matplotlib.pyplot as plt

def show_images(imgs, titles=[], n_cols=3):
    n_imgs = len(imgs)
    n_titles = len(titles)
    n_rows = np.ceil(n_imgs / n_cols)

    if n_imgs != n_titles:
        titles = [""] * n_imgs

    fig = plt.figure()
    
    for i, (img, title) in enumerate(zip(imgs, titles)):
        fig.add_subplot(n_rows, n_cols, i+1)
        plt.imshow(img, cmap='binary')
        plt.title(title)
    
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

