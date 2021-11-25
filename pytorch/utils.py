import os.path as osp

Config = {}
Config ={}
Config["root_path"] = "../polyvore_outfits_hw/polyvore_outfits"
Config["meta_file"] = osp.join(Config["root_path"], "polyvore_item_metadata.json")
Config["category_file_path"] = osp.join(Config["root_path"], "categories.csv")
Config["checkpoint_path"] = ""


Config["use_cuda"] = True
Config["debug"] = True
Config["num_epochs"] = 20
Config["batch_size"] = 64
Config["channels"] = 3

Config["learning_rate"] = 0.001
Config["num_workers"] = 5
