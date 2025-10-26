import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

"""
template = ['a photo of a {}, a type of aircraft.']
"""
template = ['a photo of a {}.']
class FGVCAircraft(DatasetBase):

    dataset_dir = 'fgvc_aircraft'

    def __init__(self, root, num_shots, setting="standard", seed=1):
        assert setting in ("standard", "base2new")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        self.template = template

        classnames = []
        with open(os.path.join(self.dataset_dir, 'variants.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, 'images_variant_train.txt')
        val = self.read_data(cname2lab, 'images_variant_val.txt')
        test = self.read_data(cname2lab, 'images_variant_test.txt')

        self.variant_to_family, self.variant_to_family_idx = self.variant_to_family_mapping()
        self.family_idx = [self.variant_to_family_idx[c] for c in classnames]
        
        # make it a few-shot dataset by subsampling both the training and validation splits 
        # NOTE: these will be overwritten by line 33, but leaving this here s.t. you know how to create
        # a new FSL dataset on the fly if needed
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # load the preprocessed jsonl for the few-shot split according to the (seed, shots) combo
        # NOTE: u should download the shared pickles on drive first
        preprocessed_train = os.path.join(self.dataset_dir, "split_fewshot", f"shot_{num_shots}-seed_{seed}_train.jsonl")
        train = OxfordPets.load_preprocessed_jsonl(preprocessed_train)
        preprocessed_val = os.path.join(self.dataset_dir, "split_fewshot", f"shot_{num_shots}-seed_{seed}_val.jsonl")
        val = OxfordPets.load_preprocessed_jsonl(preprocessed_val)

        # subsample the classes in the sets into base/novel 
        # (for the test set, we return both base/novel test loader to compute both accs in a single script)
        if setting == "base2new":
            train, val, test_base, test_new = OxfordPets.base2new_split(train, val, test)
        elif setting == "standard":
            train, val, test_base, test_new = train, val, test, None        

        super().__init__(train_x=train, val=val, test=test_base, test_new=test_new)
        

        print(f"Number of Train, Val and Test classes = {len(self.classnames)}, {len(self.val_classnames)} and {len(self.test_classnames)}")
        if setting == "base2new":
            assert all([ct == cv for ct, cv in zip(self.classnames, self.val_classnames)]) # ensure train and val classes are the same
            assert not any([ct == cv for ct, cv in zip(self.classnames, self.test_new_classnames)]) # ensure train and test classes are different
    
    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                imname = line[0] + '.jpg'
                classname = ' '.join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)
        
        return items
    
    def variant_to_family_mapping(self):
        # 1) variant 정보 읽기: id -> variant
        variant_dict = {}
        with open(self.dataset_dir + "/images_variant_test.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue  # 빈 줄 무시
                img_id = parts[0]
                variant = " ".join(parts[1:])  # id 제외 나머지를 전부 variant로 합침
                variant_dict[img_id] = variant

        # 2) family 정보 읽어서 variant -> family 매핑 생성
        variant_to_family = {}
        with open(self.dataset_dir + "/images_family_test.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_id = parts[0]
                family = " ".join(parts[1:])
                if img_id in variant_dict:
                    variant = variant_dict[img_id]
                    variant_to_family[variant] = family

        unique_families = sorted(list(set(variant_to_family.values())))

        # family -> numeric label 매핑
        family_to_idx = {fam: i for i, fam in enumerate(unique_families)}
        print(f"family len: {len(family_to_idx)}")

        # variant -> numeric family label 변환
        variant_to_family_idx = {var: family_to_idx[fam] 
                                for var, fam in variant_to_family.items()}

        
        return variant_to_family, variant_to_family_idx
