import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img


frame = 345
# Read a png image file (RGB)
img_L = img.imread('/Users/yidu/Desktop/UB_RA_Works/Super_Map/Data/P000/image_left/000' + str(frame) + '_left.png')
img_R = img.imread('/Users/yidu/Desktop/UB_RA_Works/Super_Map/Data/P000/image_right/000' + str(frame) + '_right.png')
# Read a depth image (D)
img_depth_L = np.load('/Users/yidu/Desktop/UB_RA_Works/Super_Map/Data/P000/depth_left/000' + str(frame) + '_left_depth.npy')
img_depth_R = np.load('/Users/yidu/Desktop/UB_RA_Works/Super_Map/Data/P000/depth_right/000' + str(frame) + '_right_depth.npy')


print(img_L.shape)
print(img_depth_L.shape)

print(img_depth_L.shape[0]*img_depth_L.shape[0])


# region image show
# fig, axs = plt.subplots(2, 2)
# fig.suptitle('Frame - ' + str(frame))
# # show image
# axs[0, 0].imshow(img_L), axs[0, 0].set_title('Left Image')
# axs[0, 1].imshow(img_R), axs[0, 1].set_title('Right Image')
# # show depth image
# axs[1, 0].imshow(img_depth_R, cmap='gray'), axs[1, 0].set_title('Left Depth')
# axs[1, 1].imshow(img_depth_R, cmap='gray'), axs[1, 1].set_title('Right Depth')
# fig.tight_layout()
# plt.show()
# endregion