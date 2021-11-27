import os.path as osp

Config = {}
Config ={}
Config["project_root"] = "../"
Config["root_path"] = "../polyvore_outfits_hw/polyvore_outfits"
Config["meta_file"] = osp.join(Config["root_path"], "polyvore_item_metadata.json")
Config["category_file_path"] = osp.join(Config["root_path"], "categories.csv")
Config["compatibility_train"] = osp.join(Config["root_path"], "compatibility_train.txt")
Config["compatibility_valid"] = osp.join(Config["root_path"], "compatibility_valid.txt")
Config["outfits_train"] = osp.join(Config["root_path"], "train.json")
Config["outfits_valid"] = osp.join(Config["root_path"], "valid.json")
Config["checkpoint_path"] = ""


Config["use_cuda"] = True
Config["debug"] = False
Config["num_epochs"] = 20
Config["batch_size"] = 128
Config["channels"] = 3

Config["learning_rate"] = 0.001
Config["num_workers"] = 4
