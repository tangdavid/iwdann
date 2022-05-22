from data_list import ImageList, LoadedImageList
from torchvision import transforms
from pickle import dump

PAIRS = [('target_train.txt', 'target_train_dataset.pkl'),
         ('target_test.txt', 'target_test_dataset.pkl')]

for source, dest in PAIRS:
    source = f'data/iwildcam/{source}'
    dest = f'data/iwildcam/{dest}'
    dset_source = ImageList(open(source).readlines(), transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]), mode='RGB', root_folder='data/iwildcam')
    loaded_dset_source = LoadedImageList(dset_source)
    with open(dest, 'wb') as f:
        dump([loaded_dset_source.samples.numpy(),
                     loaded_dset_source.targets.numpy()], f)