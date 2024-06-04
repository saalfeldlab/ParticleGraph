from GNN_particles_Ntype import *

if __name__ == '__main__':
    im = imread(f"graphs_data/ABCD.tif")
    # im = np.rot90(im, k=3, axes=(1, 2))
    im = im / 255*0.5
    video = np.zeros((101, 100,100))
    for k in trange (101):
        for ax in range(4):
            print (-1+2*(ax%2))
            im[ax] = np.roll(im[ax], -1 + 2*(ax//2), axis=ax%2)
            video[k] = np.sum(im, axis=0)

    imsave("graphs_data/ABCD_video.tif", video.astype(np.float32))




