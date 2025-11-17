'''
Copyright 2025 Sony Semiconductor Solutions Corporation. This is UNPUBLISHED PROPRIETARY SOURCE
CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, 
sold, and distributed in any form or by any means without prior explicit permission in writing 
of Sony Semiconductor Solutions Corporation.
'''

import numpy as np

def get_header(scales):
    lines = [
        "# -----------------------------------------------------------",
        "# Anchor Box data for the Sony Detection model",
        "# -----------------------------------------------------------",
        "# Scales",
        "# y,  x,  h,  w",
        f"  {scales[0]}, {scales[1]}, {scales[2]}, {scales[3]},",
        "# Anchor boxes",
        "# y,  x,  h,  w"
    ]
    return "\n".join(lines) + "\n"

if __name__ == "__main__":

    # Set config import here:
    from model.config.generic_ssd_config import priors,center_variance,size_variance
    
    # Windows file paths for this part
    # Set output file location to write the anchor boxes:
    out_file = r'..\..\data\anchor_boxes_pt.txt'

    scales = [center_variance**-1,center_variance**-1,size_variance**-1,size_variance**-1] # Sy, Sx, Sh, Sw
    scales_f = [str(int(x)) for x in scales]

    anchors_raw = priors.numpy()
    
    anchors_y = anchors_raw[:, 0]
    anchors_x = anchors_raw[:, 1]
    anchors_h = anchors_raw[:, 2]
    anchors_w = anchors_raw[:, 3]

    anchors_y = np.round(anchors_y * (1 << 7))
    anchors_x = np.round(anchors_x * (1 << 7))
    anchors_h = np.round(anchors_h * (1 << 7))
    anchors_w = np.round(anchors_w * (1 << 7))

    with open(out_file, 'w') as f:
        header = get_header(scales_f)
        f.write(header)
        for i in range(0, len(anchors_raw)):
            f.write('  {}, {}, {}, {},\n'.format(
                # yc,xc,h,w
                int(anchors_y[i]), int(anchors_x[i]), int(anchors_h[i]), int(anchors_w[i]))
                )

    # # Set config import here:
    # from model.config.generic_ssd_config import priors,center_variance,size_variance

    # # Set output file location to write the anchor boxes:
    # out_file = r'..\..\data\anchor_boxes_pt.txt'

    # scales = [center_variance**-1,center_variance**-1,size_variance**-1,size_variance**-1] # Sy, Sx, Sh, Sw
    # scales_f = [str(int(x)) for x in scales]

    # anchors_raw = priors.numpy()
    
    # anchors_x = anchors_raw[:, 0]
    # anchors_y = anchors_raw[:, 1]
    # anchors_w = anchors_raw[:, 2]
    # anchors_h = anchors_raw[:, 3]

    # anchors_y = np.round(anchors_y * (1 << 7))
    # anchors_x = np.round(anchors_x * (1 << 7))
    # anchors_h = np.round(anchors_h * (1 << 7))
    # anchors_w = np.round(anchors_w * (1 << 7))

    # with open(out_file, 'w') as f:
    #     header = get_header(scales_f)
    #     f.write(header)
    #     for i in range(0, len(anchors_raw)):
    #         f.write('  {}, {}, {}, {},\n'.format(
    #             # yc,xc,h,w
    #             int(anchors_y[i]), int(anchors_x[i]), int(anchors_h[i]), int(anchors_w[i]))
    #             )
        