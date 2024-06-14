import os


def check_local_model_exists(model_save_paths) -> str:
    for root in model_save_paths:
        if os.path.exists(root):
            model_path = root
            return model_path
    return ""
