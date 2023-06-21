import json
import yaml
import os
import os.path as osp
import cv2
import matplotlib
# matplotlib.use('Cairo')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import cProfile, pstats
import concurrent.futures as cf
import time
from datetime import timedelta
import shapely.geometry as sg
from tqdm import tqdm
import logging
import random
from parsing.utils.visualization import ImagePlotter
import seaborn as sns
from structured3D_geometry import InvalidGeometryError
from structured3D2wireframe import InvalidRooms, generate_negative_edges, merge_json, compute_label_stats, broken_symlink
from parsing.utils.plane_generation import CycleBasisGeneration
from parsing.utils.plane_evaluation import centroid_from_polygon
import networkx as nx
import csv

PLANE_CLASSES = (
    'invalid', #Never output, just for training purposes.
    'wall',
    'floor',
    'ceiling',
)
P_ENUM = {PLANE_CLASSES[idx]:idx for idx in range(len(PLANE_CLASSES))}

JUNCTION_CLASSES = (
    'invalid',
    'false',
    'proper'
)
J_ENUM = {JUNCTION_CLASSES[idx]:idx for idx in range(len(JUNCTION_CLASSES))}
LINE_CLASSES = (
    'invalid',
    'valid'
)


P_IDX = 0
NBR_NEG_PLANES = 20


DATA_RANGE = {
    'train':(0,3000),
    'val':(3000, 3250),
    'test':(3250, 3500)
}

def link_image(s3d_img_path, out_image_dir, img_name):
    try:
        #Remove if existing
        os.remove(osp.abspath(osp.join(out_image_dir, img_name)))
    except IOError:
        pass
    os.symlink(
        osp.relpath(s3d_img_path, start = out_image_dir),
        osp.join(out_image_dir, img_name))

def link_and_annotate(root, scene_dir, out_image_dir, make_plot = False, invalid_rooms = None):
    logger = logging.getLogger('structured3D2wireframe')
    scene_id = scene_dir.split('_')[1]
    if invalid_rooms.scene_is_invalid(scene_id):
        raise InvalidGeometryError('Scene is on invalid list')

    out_ann = []
    plot_scene_dir = '/host_home/plots/planes_simple/{}'.format(scene_id)
    if make_plot:
        os.makedirs(plot_scene_dir, exist_ok=True)


    # print('Scene', scene_id)
    render_dir = osp.join(root, scene_dir, '2D_rendering')
    for room_id in os.listdir(render_dir):
        # print('Room', room_id)
        if invalid_rooms and invalid_rooms.room_is_invalid(scene_id, room_id):
            logger.info('Skipping Room {} in Scene {}, it is invalid'.format(room_id, scene_id))
            continue
        room_dir = osp.join(render_dir, room_id, 'perspective', 'full')
        for pos_id in os.listdir(room_dir):
            ann = {}
            pos_dir = osp.join(room_dir, pos_id)
            # RGB
            img_name = 'S{}R{:0>5s}P{}.png'.format(scene_id, room_id, pos_id)
            rgb_src_path = osp.join(pos_dir,'rgb_rawlight.png')
            img = cv2.imread(rgb_src_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ann['filename'] = img_name
            ann['height'], ann['width'] = img.shape[:2]
            link_image(rgb_src_path, out_image_dir, img_name)

            # Depth
            depth_img_name = 'S{}R{:0>5s}P{}_depth.png'.format(scene_id, room_id, pos_id)
            depth_src_path = osp.join(pos_dir,'depth.png')
            ann['filename_depth'] = depth_img_name
            link_image(depth_src_path, out_image_dir, depth_img_name)


            with open(osp.join(pos_dir, 'layout.json')) as f:
                layout_ann = json.load(f)

            with open(osp.join(pos_dir, 'camera_pose.txt')) as f:
                csvreader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                #vx vy vz tx ty tz ux uy uz xfov yfov 1
                camera_pose = next(csvreader)

            plane_ann = get_plane_annotation_from_S3D(img, layout_ann, camera_pose, scene_id)
            if plane_ann is None:
                logger.warning('No junctions found for image {}, skipping'.format(img_name))
                continue
            for p in plane_ann['planes']:
                if p['junction_idx'][0] != p['junction_idx'][-1]:
                    # os.makedirs(plot_scene_dir, exist_ok=True)
                    # plotter = ImagePlotter(LINE_CLASSES, JUNCTION_CLASSES, PLANE_CLASSES)
                    # plane_ann.update(ann)
                    # plotter.plot_gt_image(img, plane_ann, plot_scene_dir, desc = 'pos')#, edges_text = edge2l_idx)
                    logger.warning('Incorrect polygon for image {}, skipping'.format(img_name))
                    continue

            ann.update(plane_ann)

            if make_plot:
                plotter = ImagePlotter(LINE_CLASSES, JUNCTION_CLASSES, PLANE_CLASSES)
                plotter.plot_gt_image(img, ann, plot_scene_dir, desc = 'pos')#, edges_text = edge2l_idx)
                plotter.plot_gt_image(img, ann, plot_scene_dir, desc = 'neg_edge', use_negative_edges = True)
                plotter.plot_gt_image(img, ann, plot_scene_dir, desc = 'neg_plane', use_negative_planes = True)

            out_ann.append(ann)

    return out_ann

def get_plane_annotation_from_S3D(image, layout_ann, camera_pose, scene_id=None):
    out_junctions = []
    out_junctions_semantic = []
    out_edges = []
    out_edges_semantic = []
    out_planes = []
    height, width = image.shape[:2]

    # Remove not visible junctions
    visible_ids = set()
    for p in layout_ann['planes']:
        for poly in p['visible_mask']:
            visible_ids |= set(poly)

    visible_ids = sorted(visible_ids)
    inverse_idx = {v_idx:idx for (idx,v_idx) in enumerate(visible_ids)}
    # print('visible_ids',visible_ids)
    # print('inverse_idx',inverse_idx)
    # Filter
    layout_ann['junctions'] = [layout_ann['junctions'][idx] for idx in visible_ids]
    for p in layout_ann['planes']:
        new_mask_ids = []
        for poly in p['visible_mask']:
            new_mask_ids.append([inverse_idx[p_idx] for p_idx in poly])
        p['visible_mask'] = new_mask_ids


    t = time.time()
    for j in layout_ann['junctions']:
        out_junctions.append(j['coordinate'])
        if (j['ID'] is None) or (not j['isvisible']):
            state = J_ENUM['false']
        else:
            state  = J_ENUM['proper']
        out_junctions_semantic.append(state)

    junction2edge = -np.ones([len(out_junctions),len(out_junctions)], dtype=np.int32)
    out_junctions_np = np.array(out_junctions)
    out_edges = []

    generated_planes = set()

    for p in layout_ann['planes']:
        for poly in p['visible_mask']:
            # Find edges for polygon
            p_edges = []
            # if poly[0] != poly[-1]:
            #     raise InvalidGeometryError('Polygon is not complete')
            fr_poly = frozenset(poly)
            assert fr_poly not in generated_planes

            for edge in zip(poly[:-1],poly[1:]):
                edge = sorted(list(edge))

                # Skip self loops in polygon
                if edge[0] == edge[1]:
                    continue
                assert edge[0] != edge[1]

                e_idx = int(junction2edge[edge[0],edge[1]])
                if e_idx < 0:
                    #Edge does not exist, create it
                    e_idx = len(out_edges)
                    junction2edge[edge[0],edge[1]] = e_idx
                    out_edges.append(edge)
                p_edges.append(e_idx)

            # Check self loops in edges
            for e0, e1 in zip(p_edges[:-1],p_edges[1:]):
                assert e0 != e1

            centroid_np = centroid_from_polygon(out_junctions_np[poly])
            plane = {
                'junction_idx': poly,
                'edge_idx': p_edges,
                'semantic': P_ENUM[p['type']],
                'parameters': p['normal'] + [p['offset']],
                'centroid': centroid_np.tolist()
            }
            out_planes.append(plane)

            generated_planes.add(fr_poly)


    if len(out_junctions) > 0:
        out = {}
        out_junctions_np = np.array(out_junctions)
        out_edges_np = np.array(out_edges)
        out['junctions'] = out_junctions
        out['edges_positive'] = out_edges
        out['junctions_semantic'] = out_junctions_semantic
        out['edges_semantic'] = [1] * len(out_edges)
        out['edges_all_semantic'] = out['edges_semantic']
        out['edges_negative'] = generate_negative_edges(image, out_junctions_np, out_edges_np)
        out['planes'] = out_planes
        out['camera_pose'] = camera_pose
        # t = time.time()

        cg = CycleBasisGeneration(np.array(out_edges+out['edges_negative'][:10]), out_junctions_np)
        neg_node_idx, neg_edge_idx = cg.generate_planes(number_of_planes = NBR_NEG_PLANES, return_edge_idx = True)
        out['planes_negative'] = [{
            'junction_idx': poly.tolist(),
            'edge_idx': poly_edges.tolist(),
            'semantic': 0,
            'centroid': centroid_from_polygon(out_junctions_np[poly,:]).tolist()
            } for poly, poly_edges in zip(neg_node_idx, neg_edge_idx) ]

        # out['planes_negative'] = generate_negative_planes(out_planes, out['edges_negative'], out_junctions_np)
        # print('Negative planes', time.time()-t)
    else:
        out = None

    return out

def generate_negative_planes(planes, edges, junctions, junction2edge = None):
    # t = time.time()
    G = nx.Graph()
    G.add_nodes_from(range(junctions.shape[1]))
    G.add_edges_from(edges)
    di_G = G.to_directed()
    cycles = nx.simple_cycles(di_G)

    # print('=================================================================', time.time()-t)
    # print('Find cycles', time.time()-t)
    planes_negative = []
    plane_cycles = PlaneCycles(planes)
    # polygon_time = 0
    # cycle_time = 0
    # t_loop = time.time()
    while len(planes_negative) < NBR_NEG_PLANES:
        try:
            cycle_sample = next(cycles)
        except StopIteration:
            break

        if len(cycle_sample) < 3:
            # Polygon must have 3 nodes.
            continue

        # t = time.time()
        cycle_exists = plane_cycles.cycle_exists(cycle_sample)
        # cycle_time += time.time()-t
        if cycle_exists:
            continue
        poly = cycle_sample + [cycle_sample[0]]

        # Verify its a valid polygon
        # t = time.time()
        sg_poly = sg.Polygon(junctions[np.array(poly)])
        # polygon_time += time.time()-t
        if not sg_poly.is_valid:
            continue


        if junction2edge is not None:
            poly_edges = []
            for edge in zip(poly[:-1],poly[1:]):
                edge = sorted(list(edge))
                e_idx = int(junction2edge[edge[0],edge[1]])
                poly_edges.append(e_idx)
        else:
            poly_edges = []

        planes_negative.append({
            'junction_idx': poly,
            'edge_idx': poly_edges,
            'semantic': 0
        })
    # print('Cycle time',cycle_time)
    # print('Polygon time',polygon_time)
    # print('Loop time',time.time()-t_loop)
    return planes_negative





class PlaneCycles:
    # Plane polygons have identical last and first nodes
    # In cycles each node occur only one.
    def __init__(self, planes):
        self.plane_cycles = [
            np.array(p['junction_idx'][:-1], dtype=np.uint) for p in planes
        ]
        self.plane_cycles_flipped = [
            np.flip(p) for p in self.plane_cycles
        ]

    def _cycle_equal(self, p, cycle):
        first_idx = np.flatnonzero(cycle[0]==p)
        # Roll cycle to match with element in p
        rolled_equal = np.roll(cycle,first_idx[0]) == p
        # All elements equal, found equal cycle
        return np.all(rolled_equal)

    def cycle_exists(self, cycle):
        for p, p_flipped in zip(self.plane_cycles, self.plane_cycles_flipped):
            if not len(p) == len(cycle):
                # Incorrect length
                continue

            if not np.any(cycle[0]==p):
                # Could not find the node
                continue
            cycle = np.array(cycle, dtype=np.uint)

            if self._cycle_equal(p,cycle) or self._cycle_equal(p_flipped,cycle):
                return True
        return False



def generate_sub_ann(data_dir, scene_dir, out_image_dir, out_json_dir, make_plot=False, invalid_rooms = None):
    json_path = osp.join(out_json_dir, scene_dir)
    if not osp.exists(json_path) or broken_symlink(out_image_dir, json_path):
        generated = True
        ann = link_and_annotate(data_dir, scene_dir, out_image_dir, make_plot=make_plot, invalid_rooms = invalid_rooms)
        with open(json_path, 'w') as f:
            json.dump(ann, f)
    else:
        generated = False

    return generated


if __name__ == '__main__':
    script_path = osp.dirname(osp.realpath(__file__))
    parser = argparse.ArgumentParser(description='Generate wireframe format from Structured3D', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=str, help='Path to Structured3D')
    parser.add_argument('out_dir', type=str, help='Path to storing conversion')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')
    parser.add_argument('-s', '--nbr-scenes', type=int, default = None, help='Number of scenes to process')
    parser.add_argument('-l', '--logfile', type=str, default = None, help='logfile path if wanted')
    parser.add_argument('--invalid', type=str, help='Invalid list from Structured3D',
                        default = osp.abspath(osp.join(script_path, '..', 'libs', 'Structured3D', 'metadata', 'errata.txt')))
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite out_dir if existing')
    parser.add_argument('--halt', action='store_true', help='Halt on error')
    parser.add_argument('--merge-only', action='store_true', help='Only do merge, skip looking for new annotations')
    parser.add_argument('--profile', action='store_true', help='Run profiler on one scene')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot images with GT lines')

    args = parser.parse_args()

    #Supress shapely logger
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    # create logger
    logger = logging.getLogger('structured3D2wireframe')
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if args.logfile:
        fh = logging.FileHandler(args.logfile, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    invalid_rooms = InvalidRooms(args.invalid) if args.invalid else None

    if osp.exists(args.out_dir) and not args.overwrite:
        print("Output directory {} already exists, specify -o flag if overwrite is permitted".format(args.out_dir))
        sys.exit()

    out_image_dir = osp.join(args.out_dir, 'images')
    os.makedirs(out_image_dir, exist_ok = True)
    out_json_dir = osp.join(args.out_dir, 'ann')
    os.makedirs(out_json_dir, exist_ok = True)

    dirs = (os.listdir(args.data_dir))

    if args.profile:
        pr= cProfile.Profile()
        pr.enable()
        link_and_annotate(args.data_dir, dirs[0], out_image_dir, make_plot=False)
        pr.disable()
        with open('stats.profile', 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
            ps.print_stats()
        sys.exit()

    if args.nbr_scenes:
        dirs = dirs[:args.nbr_scenes]

    if args.merge_only:
        dirs = []

    futures = []
    start = time.time()
    nbr_failed = 0
    nbr_invalid = 0
    nbr_generated = 0
    nbr_existed = 0

    with cf.ProcessPoolExecutor(max_workers = args.nbr_workers) as executor:
        for scene_dir in dirs:
            # if not scene_dir.endswith('00173'):
            #     continue
            f_args = (args.data_dir, scene_dir, out_image_dir, out_json_dir)
            f_kwargs = dict(make_plot=args.plot,
                            invalid_rooms = invalid_rooms)
            f = executor.submit(generate_sub_ann, *f_args, **f_kwargs)
            futures.append(f)

        #Wait for results, waits for processes to finish and raises errors
        for i, f in enumerate(tqdm(cf.as_completed(futures), total = len(futures))):
            try:
                generated = f.result()
                nbr_generated += generated
                nbr_existed += not generated
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                executor.shutdown()
            except InvalidGeometryError:
                logger.exception('Invalid geometry for scene {}'.format(dirs[i]))
                nbr_invalid += 1
            except:
                logger.exception('Got exception for scene {}'.format(dirs[i]))
                nbr_failed += 1
                if args.halt:
                    for f in futures:
                        f.cancel()
                    executor.shutdown()
                    raise

    logger.info('Generating {} scenes took {}'.format(len(dirs), timedelta(seconds=time.time()-start)))
    logger.info('{}/{} scenes generated'.format(nbr_generated, len(dirs)))
    logger.info('{}/{} scenes existed'.format(nbr_existed, len(dirs)))
    logger.info('{}/{} scenes failed'.format(nbr_failed, len(dirs)))
    logger.info('{}/{} scenes invalid'.format(nbr_invalid, len(dirs)))


    start = time.time()
    ann, nbr_scenes_split, nbr_images_split = merge_json(out_json_dir, args.out_dir)
    nbr_scenes_merged = np.sum([v for k,v in nbr_scenes_split.items()])
    logger.info('Merging {} scenes took {}'.format(nbr_scenes_merged,  timedelta(seconds=time.time()-start)))
    logger.info('Scene split is:')
    for k,v in nbr_scenes_split.items():
        print(k,':',v)

    nbr_images = len(ann)
    r = {}
    r['nbr_junctions'] = np.array([len(a['junctions']) for a in ann])
    r['nbr_visible_junctions'] = np.array([(np.array(a['junctions_semantic'], dtype=int) == 1).sum() for a in ann])
    r['nbr_edges_pos'] = np.array([len(a['edges_positive']) for a in ann])
    r['nbr_edges_neg'] = np.array([len(a['edges_negative']) for a in ann])

    fig, ax = plt.subplots(2,2)
    for i,title in enumerate(r):
        ax1 = ax.flat[i]
        ax1.hist(r[title], bins=15)
        ax1.set_title(title)
        ax1.set_ylabel('Nbr images (total: {})'.format(nbr_images))
        ax1.set_xlabel('{} / image'.format(title))

    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, 'stats.svg'))
    plt.close()


    line_c = np.concatenate([np.array(a['edges_all_semantic'], dtype=np.int32) for a in ann])

    fig, ax = plt.subplots(2,1)
    stats_all_labels = compute_label_stats(line_c, LINE_CLASSES, ax=ax[0])
    stats_all_labels['nbr_images_split'] = nbr_images_split
    stats_all_labels['nbr_scenes_split'] = nbr_scenes_split

    with open(osp.join(args.out_dir, 'stats_all.yaml'), 'w') as f:
        yaml.safe_dump(stats_all_labels, f, default_flow_style=None)

    line_c = np.concatenate([np.array(a['edges_semantic'], dtype=np.int32) for a in ann])
    stats_simple_labels = compute_label_stats(line_c,LINE_CLASSES, ax=ax[1])

    with open(osp.join(args.out_dir, 'stats_simple.yaml'), 'w') as f:
        yaml.safe_dump(stats_simple_labels, f, default_flow_style=None)

    plt.tight_layout()
    plt.savefig(osp.join(args.out_dir, 'label_stats.svg'))
    plt.close()
