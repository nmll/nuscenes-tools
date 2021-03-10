
# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

import trk_render as render

import argparse
import json
import os
import time
from typing import Tuple, List, Dict, Any, Callable, Tuple
import tqdm
import sklearn
import numpy as np
import unittest

try:
    import pandas
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as pandas was not found!')


from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData
from nuscenes.eval.tracking.loaders import create_tracks
#from nuscenes.eval.tracking.render import TrackingRenderer, recall_metric_curve, summary_plot
from trk_render import TrackingRenderer, recall_metric_curve, summary_plot
from nuscenes.eval.tracking.utils import print_final_metrics
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
from nuscenes.utils.data_classes import LidarPointCloud#读取点云


save_path = "./visresult/pp_bbox_default/"      #保存路径及名称
datset_path = "/share/nuscenes/v1.0-trainval/"  #数据集路径
#result_file = "/share/OpenPCDet/output/cfgs/nuscenes_models/cbgs_second_multihead/sec_neiborcbam_reid/eval/epoch_20/val/default/\
#final_result/data/trk_results/trk_results_nusc.json"

result_file = "/share/OpenPCDet/output/cfgs/nuscenes_models/cbgs_pp_multihead/default/eval/epoch_5823/val/default/\
final_result/data/trk_results/trk_results_nusc.json"  #结果文件
#result_file = "/share/OpenPCDet/output/cfgs/nuscenes_models/cbgs_second_multihead/default/eval/epoch_6229/val/default/final_result/data/\
#trk_results/trk_results_nusc.json"
#info_path = datset_path + "/nuscenes_infos_10sweeps_val.pkl"
render_classes = ["car"]      #所需要画的种类
frame_id_thr = 30 #no use   
ifplotgt = False            
scene_id_thr = 150         #共150个场景，每个场景越40帧 仅画前 scene_id_thr 个场景
lidar_name = 'LIDAR_TOP'
cam_name = 'CAM_FRONT'


class TrackingEval:
    """
    This is the official nuScenes tracking evaluation code.
    Results are written to the provided output_dir.
    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.
    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.
    Please see https://www.nuscenes.org/tracking for more details.
    """
    def __init__(self,
                 config: TrackingConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str,
                 nusc_version: str,
                 nusc_dataroot: str,
                 verbose: bool = True,
                 render_classes: List[str] = None):
        """
        Initialize a TrackingEval object.
        :param config: A TrackingConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param nusc_version: The version of the NuScenes dataset.
        :param nusc_dataroot: Path of the nuScenes dataset on disk.
        :param verbose: Whether to print to stdout.
        :param render_classes: Classes to render to disk or None.
        """
        self.cfg = config
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.render_classes = render_classes

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Initialize NuScenes object.
        # We do not store it in self to let garbage collection take care of it and save memory.
        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)
        self.nusc = nusc
        # Load data.
        if verbose:
            print('Initializing nuScenes tracking evaluation')
        pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, TrackingBox,
                                                verbose=verbose)
        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        # Add center distances.
        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering tracks')
        pred_boxes = filter_eval_boxes(nusc, pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth tracks')
        gt_boxes = filter_eval_boxes(nusc, gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = gt_boxes.sample_tokens   #len():6019

        # Convert boxes to tracks format.
        self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)


    def evaluate(self) -> Tuple[TrackingMetrics, TrackingMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        metrics = TrackingMetrics(self.cfg)

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = TrackingMetricDataList()

        def accumulate_class(curr_class_name):
            # curr_ev = TrackingEvaluation(self.tracks_gt, self.tracks_pred, curr_class_name, self.cfg.dist_fcn_callable,\
            #                              self.cfg.dist_th_tp, self.cfg.min_recall,\
            #                              num_thresholds=TrackingMetricData.nelem,\
            #                              metric_worst=self.cfg.metric_worst,\
            #                              verbose=self.verbose,\
            #                              output_dir=self.output_dir,\
            #                              render_classes=self.render_classes)
            #curr_md = curr_ev.accumulate()
            
            """
            Compute metrics for all recall thresholds of the current class.
            :return: TrackingMetricData instance which holds the metrics for each threshold.
            """
            # Init.
            if self.verbose:
                print('Computing metrics for class %s...\n' % curr_class_name)
            accumulators = []
            thresh_metrics = []
            #md = TrackingMetricData()

            # Skip missing classes.
            gt_box_count = 0
            gt_track_ids = set()
            for scene_tracks_gt in self.tracks_gt.values():
                for frame_gt in scene_tracks_gt.values():
                    for box in frame_gt:
                        if box.tracking_name == curr_class_name:
                            gt_box_count += 1
                            gt_track_ids.add(box.tracking_id)
            if gt_box_count == 0:
                print("gtboxcount=0")
                # Do not add any metric. The average metrics will then be nan.
                #return md

            # Register mot metrics.
            #mh = create_motmetrics()

            # Get thresholds.
            # Note: The recall values are the hypothetical recall (10%, 20%, ..).
            # The actual recall may vary as there is no way to compute it without trying all thresholds.
            thresholds = np.array([0.1])  #, recalls = self.compute_thresholds(gt_box_count)
            #md.confidence = thresholds
            #md.recall_hypo = recalls
            if self.verbose:
                print('Computed thresholds\n')

            for t, threshold in enumerate(thresholds):
                # If recall threshold is not achieved, we assign the worst possible value in AMOTA and AMOTP.
                if np.isnan(threshold):
                    continue

                # Do not compute the same threshold twice.
                # This becomes relevant when a user submits many boxes with the exact same score.
                if threshold in thresholds[:t]:
                    continue
                    """
                        Accumulate metrics for a particular recall threshold of the current class.
                        The scores are only computed if threshold is set to None. This is used to infer the recall thresholds.
                        :param threshold: score threshold used to determine positives and negatives.
                        :return: (The MOTAccumulator that stores all the hits/misses/etc, Scores for each TP).
                    """
                accs = []
                scores = []  # The scores of the TPs. These are used to determine the recall thresholds initially.

                # Go through all frames and associate ground truth and tracker results.
                # Groundtruth and tracker contain lists for every single frame containing lists detections.
                tracks_gt = self.tracks_gt
                scene_num_id = 0
                for scene_id in tqdm.tqdm(list(tracks_gt.keys()), disable=not self.verbose, leave=False):#按场景

                    # Initialize accumulator and frame_id for this scene
                    acc = MOTAccumulatorCustom()
                    frame_id = 0  # Frame ids must be unique across all scenes
                    

                    # Retrieve GT and preds.
                    scene_tracks_gt = tracks_gt[scene_id]
                    scene_tracks_pred = self.tracks_pred[scene_id]
                    # if len(tracks_gt) == 151:
                    #     tracks_gt.pop('0')        
                    # Visualize the boxes in this frame.
                    if curr_class_name in self.render_classes and threshold is not None and scene_num_id < scene_id_thr:
                    
                        save_path = os.path.join(self.output_dir, 'render', str(scene_id), curr_class_name)
                        os.makedirs(save_path, exist_ok=True)
                        renderer = TrackingRenderer(save_path)
                    else:
                        renderer = None

                    for timestamp in scene_tracks_gt.keys(): #每个场景分别每帧
                        # Select only the current class.
                        frame_gt = scene_tracks_gt[timestamp]
                        frame_pred = scene_tracks_pred[timestamp]
                        frame_gt = [f for f in frame_gt if f.tracking_name == curr_class_name]
                        frame_pred = [f for f in frame_pred if f.tracking_name == curr_class_name]

                        # Threshold boxes by score. Note that the scores were previously averaged over the whole track.
                        if threshold is not None:
                            frame_pred = [f for f in frame_pred if f.tracking_score >= threshold]

                        # Abort if there are neither GT nor pred boxes.
                        gt_ids = [gg.tracking_id for gg in frame_gt]
                        pred_ids = [tt.tracking_id for tt in frame_pred]
                        if len(gt_ids) == 0 and len(pred_ids) == 0:
                            continue

                        # Calculate distances.     
                        # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
                        assert self.cfg.dist_fcn_callable.__name__ == 'center_distance'
                        if len(frame_gt) == 0 or len(frame_pred) == 0:    
                            distances = np.ones((0, 0))
                        else:
                            gt_boxes = np.array([b.translation[:2] for b in frame_gt])
                            pred_boxes = np.array([b.translation[:2] for b in frame_pred])
                            distances = sklearn.metrics.pairwise.euclidean_distances(gt_boxes, pred_boxes)

                        # Distances that are larger than the threshold won't be associated.
                        assert len(distances) == 0 or not np.all(np.isnan(distances))
                        distances[distances >= self.cfg.dist_th_tp] = np.nan

                        # Accumulate results.
                        # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                        acc.update(gt_ids, pred_ids, distances, frameid=frame_id)
                       
                        # Store scores of matches, which are used to determine recall thresholds.
                        if threshold is not None:
                            events = acc.events.loc[frame_id]
                            matches = events[events.Type == 'MATCH']
                            match_ids = matches.HId.values
                            match_scores = [tt.tracking_score for tt in frame_pred if tt.tracking_id in match_ids]
                            scores.extend(match_scores)
                        else:
                            events = None
                

                        
                        
                        # Render the boxes in this frame.
                        if curr_class_name in self.render_classes and threshold is not None and scene_num_id < scene_id_thr:
                            # load lidar points  data按每帧加载
                            #https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_kitti.py
                            try:
                                frame0 = frame_pred[0]
                            except:
                                frame0 = scene_tracks_gt[timestamp][0]
                            sample = self.nusc.get('sample', frame0.sample_token) #frame_pred是该帧所有的物体
                            #sample_annotation_tokens = sample['anns'] #标注
                            #cam_front_token = sample['data'][cam_name]#某点位的图像
                            lidar_token = sample['data'][lidar_name]
                                # Retrieve sensor records.
                            #sd_record_cam = self.nusc.get('sample_data', cam_front_token)
                            sd_record_lid = self.nusc.get('sample_data', lidar_token)
                            cs_record = self.nusc.get('calibrated_sensor', sd_record_lid["calibrated_sensor_token"])
                            pose_record = self.nusc.get('ego_pose', sd_record_lid["ego_pose_token"])
                            #cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
                            #cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])
                            # Retrieve the token from the lidar.
                            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
                            # not the camera.
                            #filename_cam_full = sd_record_cam['filename']
                            filename_lid_full = sd_record_lid['filename']
                            src_lid_path = os.path.join(datset_path, filename_lid_full)
                            points = LidarPointCloud.from_file(src_lid_path)


                            #if lidar_token == "5af9c7f124d84e7e9ac729fafa40ea01" or lidar_token == "16be583c31a2403caa6c158bb55ae616":#选择特定帧 上面要设成150个场景
                            renderer.render(events, timestamp, frame_gt, frame_pred, points, pose_record, cs_record, ifplotgt)

                        # Increment the frame_id, unless there are no boxes (equivalent to what motmetrics does).
                        frame_id += 1

                    scene_num_id += 1    
                   
                    accs.append(acc)
                print("visually have done!")


                # Accumulate track data.
                #acc, _ = self.accumulate_threshold(threshold)
                #accumulators.append(acc)

                # # Compute metrics for current threshold.
                # thresh_name = self.name_gen(threshold)
                # thresh_summary = mh.compute(acc, metrics=MOT_METRIC_MAP.keys(), name=thresh_name)
                # thresh_metrics.append(thresh_summary)

                # # Print metrics to stdout.
                # if self.verbose:
                #     print_threshold_metrics(thresh_summary.to_dict())


    
        
        for class_name in self.cfg.class_names:
            accumulate_class(class_name)



    def main(self, render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: The serialized TrackingMetrics computed during evaluation.
        """
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print metrics to stdout.
        if self.verbose:
            print_final_metrics(metrics)

        # Render curves.
        if render_curves:
            self.render(metric_data_list)

        return metrics_summary


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_path', type=str, default="%s"%(result_file) , help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default=save_path,
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default=datset_path,
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the NIPS 2019 configuration will be used.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render statistic curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--render_classes', type=str, default=render_classes, nargs='+',
                        help='For which classes we render tracking results to disk.')
    args = parser.parse_args()
    
    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)
    render_classes_ = args.render_classes

    if config_path == '':
        cfg_ = config_factory('tracking_nips_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialize(json.load(_f))

    nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                             nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                             render_classes=render_classes_)
    nusc_eval.main(render_curves=render_curves_)




