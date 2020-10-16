import matplotlib.pyplot as plt
from xml_to_mask import xml_to_mask
from mask_to_xml import mask_to_xml

# mask = xml_to_mask('test_.xml', (0,0), (10000,10000), downsample=16)
# mask_to_xml('test.xml', mask, verbose=1, downsample=16)


from wsi_dataset_util import get_grid_list
points = get_grid_list('test.svs', 512, 4)
print(points)




# from wsi_dataset_util import get_wsi_patch
# slide  = 'test.svs'
# patch, mask = get_wsi_patch(slide, patch_size=1024, downsample=4, augment=1)
# plt.subplot(1,2,1)
# plt.imshow(patch)
# plt.subplot(1,2,2)
# plt.imshow(mask)
# plt.show()
