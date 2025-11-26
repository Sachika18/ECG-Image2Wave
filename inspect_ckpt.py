import torch
p='resnet_unet_best.pth'
ck=torch.load(p,map_location='cpu')
if isinstance(ck,dict) and 'state_dict' in ck:
    sd=ck['state_dict']
else:
    sd=ck
print('Total keys:', len(sd))
for i,(k,v) in enumerate(sd.items()):
    if hasattr(v,'shape'):
        print(i, k, v.shape)
    else:
        print(i, k, type(v))
    if i>=199:
        break
