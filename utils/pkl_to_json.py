import argparse
import os
import json
import pickle
import numpy as np


def main(pickle_path, output_json):
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)

    output = []

    for res in results:
        image_id = res.get('img_id')
        pred_instances = res.get('pred_instances', {})
        bboxes = pred_instances.get('bboxes')
        masks = pred_instances.get('masks')
        scores = pred_instances.get('scores')
        labels = pred_instances.get('labels')

        if any(val is None for val in [image_id, bboxes, scores, labels]):
            continue

        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        for j, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            coco_bbox = [x1, y1, width, height]

            ann_dict = {
                'image_id': int(image_id),
                'category_id': int(label)+1,
                'bbox': [float(coord) for coord in coco_bbox],
                'score': float(score),
                'area': float(width * height),
                'iscrowd': 0,
            }
            if masks is not None:
                rle = masks[j]
                rle['counts'] = rle['counts'].decode("ascii")
                ann_dict['segmentation'] = rle

            output.append(ann_dict)

    with open(output_json, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    main(
        pickle_path=args.pickle_path,
        output_json=args.output_json,
    )
