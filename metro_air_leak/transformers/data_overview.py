from typing import Dict, List

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(messages: List[Dict], *args, **kwargs):
    """
    Template code for a transformer block.

    Args:
        messages: List of messages in the stream.

    Returns:
        Transformed messages
    """
    def transform(messages: List[Dict], *args, **kwargs):
        for msg in messages:
            print(msg)

    return messages
