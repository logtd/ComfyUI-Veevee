SD1_ATTN_INJ_DEFAULTS = set([])
for idx in [1,2,3,4,5,6]:
    SD1_ATTN_INJ_DEFAULTS.add(('output', idx))

SD1_RES_INJ_DEFAULTS = set()
for idx in [3,4,6]:
    SD1_RES_INJ_DEFAULTS.add(('output', idx))

SD1_FLOW_MAP = set([('input', 0), ('input', 1), ('output', 6), ('output', 7), ('output', 8)])

SD1_OUTER_MAP = SD1_FLOW_MAP
SD1_INNER_MAP = set([])
for i in range(2, 12):
    SD1_INNER_MAP.add(('input', i))
for i in range(6):
    SD1_INNER_MAP.add(('output', i))

SD_FULL_MAP = set([])
for i in range(40):
    SD_FULL_MAP.add(('input', i))
    SD_FULL_MAP.add(('output', i))

SD1_INPUT_MAP = set([])
SD1_OUTPUT_MAP = set([])
for i in range(12):
    SD1_INPUT_MAP.add(('input', i))
    SD1_OUTPUT_MAP.add(('output', i))

# TODO get actual adj upsampler indexes
SDXL_ATTN_INJ_DEFAULTS = set([])
for idx in [10, 15, 20, 25, 30]:
    SDXL_ATTN_INJ_DEFAULTS.add(('output', idx))

SDXL_RES_INJ_DEFAULTS = set([])
for idx in [1,2,4]:
    SDXL_RES_INJ_DEFAULTS.add(('output', idx))

SDXL_FLOW_MAP = set([])
for idx in range(36):
    SDXL_FLOW_MAP.add(('input', idx))
    SDXL_FLOW_MAP.add(('output', idx))

# These are approximate
SDXL_OUTER_MAP = set([])
for i in range(30, 40):
    SDXL_OUTER_MAP.add(('output', i))
for i in range(5):
    SDXL_OUTER_MAP.add(('input', i))

SDXL_INNER_MAP =set([])
for i in range(30):
    SDXL_INNER_MAP.add(('output', i))
for i in range(5, 40):
    SDXL_INNER_MAP.add(('input', i))


SDXL_INPUT_MAP = set([])
SDXL_OUTPUT_MAP = set([])
for i in range(40):
    SDXL_INPUT_MAP.add(('input', i))
    SDXL_OUTER_MAP.add(('output', i))

MAP_TYPES = ['none', 'inner', 'outer', 'full', 'input', 'output']

SD1_MAPS = {
    'none': set(),
    'inner': SD1_INNER_MAP,
    'outer': SD1_OUTER_MAP,
    'full': SD_FULL_MAP,
    'output': SD1_OUTPUT_MAP,
    'input': SD1_INPUT_MAP,
}

SDXL_MAPS = {
    'none': set(),
    'inner': SDXL_INNER_MAP,
    'outer': SDXL_OUTER_MAP,
    'full': SD_FULL_MAP,
    'output': SDXL_OUTPUT_MAP,
    'input': SDXL_INPUT_MAP,
}
