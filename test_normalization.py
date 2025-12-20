"""Debug normalization to understand the patterns."""

# Test cases
tf2_names = [
    "aspp0/BatchNorm/aspp0/BatchNorm/beta:0",
    "xception_65/entry_flow/block1/unit_1/xception_module/separable_conv1_depthwise/BatchNorm/xception_65/entry_flow/block1/unit_1/xception_module/separable_conv1_depthwise/BatchNorm/beta:0",
    "logits/semantic/weights:0",
    "aspp0/weights:0"
]

tf1_names = [
    "aspp0/BatchNorm/beta",
    "xception_65/entry_flow/block1/unit_1/xception_module/separable_conv1_depthwise/BatchNorm/beta",
    "logits/semantic/weights",
    "aspp0/weights"
]

def _strip_tf_suffixes(name: str) -> str:
    """Remove TF tensor suffix :0"""
    if name.endswith(':0'):
        name = name[:-2]
    return name

def normalize_tf2_name(name: str) -> str:
    """
    Canonicalize TF2 variable names by stripping ':0' and removing duplicate scopes.
    """
    name = _strip_tf_suffixes(name)
    
    # Check if this is a BatchNorm variable with duplicate scope
    if '/BatchNorm/' in name:
        # Find all occurrences of /BatchNorm/
        bn_indices = []
        idx = 0
        while True:
            idx = name.find('/BatchNorm/', idx)
            if idx == -1:
                break
            bn_indices.append(idx)
            idx += 1
        
        # If we have exactly 2 occurrences, check for duplication
        if len(bn_indices) == 2:
            first_bn = bn_indices[0]
            second_bn = bn_indices[1]
            
            # Extract parts
            prefix = name[:first_bn]  # Everything before first /BatchNorm/
            middle = name[first_bn+11:second_bn]  # Between the two /BatchNorm/
            suffix = name[second_bn+11:]  # After second /BatchNorm/
            
            # Check if middle == prefix
            if middle == prefix:
                # Remove duplication
                return f"{prefix}/BatchNorm/{suffix}"
    
    return name

print("Testing normalization:\n")
for tf2, tf1 in zip(tf2_names, tf1_names):
    normalized = normalize_tf2_name(tf2)
    match = "✓" if normalized == tf1 else "✗"
    print(f"{match} TF2: {tf2}")
    print(f"  Normalized: {normalized}")
    print(f"  Expected:   {tf1}")
    print()
