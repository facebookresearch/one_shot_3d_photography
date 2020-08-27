#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import caffe2.python
import caffe2.python.workspace as ws
import cv2
import numpy as np
import os
from pathlib import Path

from visualization import visualize_depth


class DepthEstimatorCaffe2:
    def __init__(self, init_net_file: str, predict_net_file: str):
        print(f"Creating Tiefenrausch model from files...")
        print(f"  Init net: '{init_net_file}'")
        print(f"  Predict net: '{predict_net_file}'")

        self.init_net = caffe2.proto.caffe2_pb2.NetDef()
        with open(init_net_file, "rb") as finit:
            self.init_net.ParseFromString(finit.read())

        self.predict_net = caffe2.proto.caffe2_pb2.NetDef()
        with open(predict_net_file, "rb") as fpred:
            self.predict_net.ParseFromString(fpred.read())

    def estimate_depth(
        self, src_file: str, out_file: str, vis_file: str = None
    ):
        print(f"Reading image file '{src_file}'...")
        bgr_image = cv2.imread(src_file)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Downscale image
        target_max_dimension = 384
        h, w, _ = rgb_image.shape
        if h > w:
            nh = target_max_dimension
            nw = int(w * nh / h)
        else:
            nw = target_max_dimension
            nh = int(h * nw / w)

        nw -= nw % 32
        nh -= nh % 32

        rgb_image = cv2.resize(rgb_image, (nw, nh), interpolation=cv2.INTER_AREA)

        # Predict depth
        input = np.transpose(rgb_image / 255.0, (2, 0, 1))
        input = input[np.newaxis, :, :, :].astype(np.float32)
        ws.ResetWorkspace()
        # ws.FeedBlob("0", input)
        ws.FeedBlob(self.predict_net.external_input[0], input)
        ws.CreateNet(self.init_net)
        ws.CreateNet(self.predict_net)
        ws.RunNet(self.init_net.name)
        ws.RunNet(self.predict_net.name)
        output_blob = self.predict_net.external_output[0]
        output = ws.FetchBlob(output_blob)
        disparity = np.exp(output.squeeze())

        if out_file is not None:
            print(f"Writing depth file '{out_file}'...")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            np.save(out_file, disparity)

        if vis_file is not None:
            vis = visualize_depth(disparity)
            print(f"Writing visualization file '{vis_file}'...")
            os.makedirs(os.path.dirname(vis_file), exist_ok=True)
            cv2.imwrite(vis_file, vis)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--init_net", type=str, default="model/tiefenrausch_init.pb")
    parser.add_argument("--predict_net", type=str, default="model/tiefenrausch.pb")

    # Parameters for processing a single file
    parser.add_argument("--src_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--vis_file", type=str)

    # Parameters for processing a directory file
    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--vis_dir", type=str)

    args, unknown = parser.parse_known_args()

    # Print settings.
    print("Settings:")
    for k, v in vars(args).items():
        print("  %s: %s" % (str(k), str(v)))

    return args


def main():
    args = parse_args()

    depth_estimator = DepthEstimatorCaffe2(args.init_net, args.predict_net)

    if args.src_file is not None:
        depth_estimator.estimate_depth(
            args.src_file, args.out_file, args.vis_file)

    if args.src_dir is not None:
        for src_file in Path(args.src_dir).rglob("*"):
            src_file = str(src_file)

            if not os.path.isfile(src_file):
                continue

            rel_path = os.path.relpath(src_file, args.src_dir)

            out_file = os.path.join(
                args.out_dir, os.path.splitext(rel_path)[0] + ".npy")

            vis_file = None
            if args.vis_dir is not None:
                vis_file = os.path.join(
                    args.vis_dir, os.path.splitext(rel_path)[0] + ".png")

            depth_estimator.estimate_depth(src_file, out_file, vis_file)

    print("Finished.")


if __name__ == '__main__':
    main()
