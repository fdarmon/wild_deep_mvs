#  Copyright (c)
#
#  Deep MVS Gone  Wild All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#      * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#      * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#
#  Deep MVS Gone  Wild All rights reseved to Thales LAS and ENPC.
#
#  This code is freely avaible for academic use only and Provided “as is” without any warranties.
#
from evaluation.pipeline_utils import get_args, load_dataset
from evaluation import metrics, run_depthmaps, fusibile, filtering

from torch.utils.data import DataLoader
from utils.colmap_utils import depthmap_colmap, create_colmap_sparse, colmap_fusion


if __name__ == "__main__":
    args = get_args()
    print(args)

    dataset = load_dataset(args)
    dataloader = DataLoader(dataset)

    if args.colmap:
        depthmap_colmap(dataloader, args)
    else:
        run_depthmaps.run(dataloader, args)

    if args.filter and not args.debug:
        dataset.nviews = args.filter_num_views
        filtering.run(dataloader, args)

    if not args.debug:
        if args.fusion == "colmap":
            create_colmap_sparse(dataloader, args)
            colmap_fusion(dataloader, args)
        elif args.fusion == "fusibile":
            fusibile.run(dataloader, args)

        if args.compute_metrics:
            metrics.run(args)

