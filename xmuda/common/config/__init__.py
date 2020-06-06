from yacs.config import CfgNode


def purge_cfg(cfg: CfgNode):
    """Purge configuration for clean logs and logical check.
    If a CfgNode has 'TYPE' attribute, its CfgNode children the key of which do not contain 'TYPE' will be removed.
    """
    target_key = cfg.get('TYPE', None)
    removed_keys = []
    for k, v in cfg.items():
        if isinstance(v, CfgNode):
            if target_key is not None and (k != target_key):
                removed_keys.append(k)
            else:
                purge_cfg(v)
    for k in removed_keys:
        del cfg[k]
