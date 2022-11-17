from beartype import beartype


@beartype
def ground_truth(events: list[dict]):
    prev_labels = {"clicks": None, "carts": set(), "orders": set()}

    for event in reversed(events):
        event["labels"] = {label: prev_labels[label].copy()  \
                if label != 'clicks' else prev_labels[label] \
                for label in ['clicks', 'carts', 'orders'] if prev_labels[label]}
        if event["type"] == "clicks":
            prev_labels['clicks'] = event["aid"]
        if event["type"] == "carts":
            prev_labels['carts'].add(event["aid"])
        elif event["type"] == "orders":
            prev_labels['orders'].add(event["aid"])

    return events[:-1]