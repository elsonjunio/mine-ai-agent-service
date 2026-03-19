def get_nested_claim(data: dict, claim_path: str):
    keys = claim_path.split('.')
    value: dict | None = data

    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)

    return value


def get_claim(data: dict, claim_path: str):
    keys = claim_path.split('.')
    value = get_nested_claim(data, claim_path)

    keys.reverse()

    claim: dict[str, any] = {}
    for key in keys:
        if not claim:
            claim[key] = value
        else:
            claim = {key: claim}

    return claim
