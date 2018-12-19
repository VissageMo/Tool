from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

dataDir = '/devdata/Coco'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

# cats = coco.loadCats(coco.getCatIds())
# nms = [cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategorise: \n{}'.format(nms))

'''
COCO categories: 
person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant stop sign 
parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella 
handbag tie suitcase frisbee skis snowboard sports ball kite baseball bat baseball glove skateboard 
surfboard tennis racket bottle wine glass cup fork knife spoon bowl banana apple sandwich orange 
broccoli carrot hot dog pizza donut cake chair couch potted plant bed dining table toilet tv laptop 
mouse remote keyboard cell phone microwave oven toaster sink refrigerator book clock vase scissors 
teddy bear hair drier toothbrush

COCO supercategories: 
outdoor food indoor appliance sports person animal vehicle furniture accessory electronic kitchen
'''

catIds = coco.getCatIds(catNms=['person', 'car', 'traffic'])
# two types of get Img:
# catIds-get all pictures of given catNms / imgIds-get picture of given number
imgIds = coco.getImgIds(catIds=catIds)
# imgIds = coco.getImgIds(imgIds=[324158])
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()

plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()
