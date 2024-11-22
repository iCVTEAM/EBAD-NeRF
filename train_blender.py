import os
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from nerf import *
import optimize_pose_linear, optimize_pose_cubic, optimize_pose_event
import torchvision.transforms.functional as torchvision_F
from load_llff import pose_interpolation_llff
import matplotlib.pyplot as plt

from metrics import compute_img_metric
import novel_view_test
from event_loss_helpers import *
from torch.utils.tensorboard import SummaryWriter

def train():
    parser = config_parser()
    args = parser.parse_args()
    args.ndc= not args.no_ndc
    args.bin_num = args.deblur_images - 1
    print()
    print("----------------start---------------")
    print('spline numbers: ', args.deblur_images)
    print('ndc', args.ndc)
    print('spherify', args.spherify)



    # Load Data
    print("-------------Loading Data-----------")
    K = None
    if args.dataset_type == 'llff':
        images_colmap, poses_colmap, bds_start, render_poses = load_llff_data(args.datadir, pose_state=None,
                                                                      factor=args.factor, recenter=True,
                                                                      bd_factor=.75, spherify=args.spherify)
        hwf = poses_colmap[0, :3, -1]

        # split train/val/test
        i_test = torch.arange(0, images_colmap.shape[0], args.llffhold)
        i_train = torch.Tensor([i for i in torch.arange(int(images_colmap.shape[0])) if (i not in i_test)]).long()

        # load initial poses and corresponding blurry images for training
        images = images_colmap[i_train]
        poses_start = poses_colmap[i_train]

        # load initial novel view poses and crresponding ground truth images for testing
        # We select 5 of the views for testing
        poses_novel = poses_colmap[i_test]
        imgs_gt_novel = load_imgs(os.path.join(args.datadir, 'images_gt_novel'))

        # initial the poses
        poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])
        poses_end_se3 = poses_start_se3
        poses_novel_se3 = SE3_to_se3_N(poses_novel[:, :3, :4])

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = torch.min(bds_start) * .9
            far = torch.max(bds_start) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    use_event = args.use_event
    if use_event:
        # load event
        event_maps = torch.load(os.path.join(args.datadir, "events.pt"))[i_train].to(device).float()
        # rgb2grey
        rgb2grey = torch.tensor([1/3, 1/3, 1/3]).to(device)
        print("Load Event Data", event_maps.shape)



    # Creating Network
    print("-----------Creating Network---------")
    basedir = args.basedir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    test_metric_file_novel = os.path.join(basedir, expname, 'test_metrics_novel.txt')
    print_file = os.path.join(basedir, expname, 'print.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    if args.load_weights:
        if args.model == "event":
            print('Event Model Loading!')
            model = optimize_pose_event.Model(poses_start_se3, poses_end_se3)
        elif args.model == "linear":
            print('Linear Spline Model Loading!')
            model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)
        elif args.model == "cubic":
            print('Cubic Spline Model Loading!')
            model = optimize_pose_cubic.Model(poses_start_se3, poses_start_se3, poses_start_se3, poses_start_se3)
        else:
            print('Wrong Model Loading!')
            return
        graph = model.build_network(args)
        optimizer, optimizer_se3 = model.setup_optimizer(args)
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(args.weight_iter))  # here
        graph_ckpt = torch.load(path)
        graph.load_state_dict(graph_ckpt['graph'])
        optimizer.load_state_dict(graph_ckpt['optimizer'])
        optimizer_se3.load_state_dict(graph_ckpt['optimizer_se3'])
        global_step = graph_ckpt['global_step']

    else:
        if args.model == "event":
            model = optimize_pose_event.Model(poses_start_se3, poses_end_se3)
        elif args.model == "linear":
            low, high = 0.0001, 0.005
            rand = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            poses_start_se3 = poses_start_se3 + rand
            model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)
        elif args.model == "cubic":
            low, high = 0.0001, 0.01
            rand1 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            rand2 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            rand3 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            poses_se3_1 = poses_start_se3 + rand1
            poses_se3_2 = poses_start_se3 + rand2
            poses_se3_3 = poses_start_se3 + rand3

            model = optimize_pose_cubic.Model(poses_start_se3, poses_se3_1, poses_se3_2, poses_se3_3)

        graph = model.build_network(args)  # nerf, nerf_fine, forward
        optimizer, optimizer_se3 = model.setup_optimizer(args)



    print("------------Start Training----------")
    N_iters = args.N_iters + 1
    writer = SummaryWriter(os.path.join(basedir, expname, 'logs'))
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)

    poses_num = images.shape[0] * args.deblur_images
    images = images.view(images.shape[0], -1, 3)

    start = 0
    if not args.load_weights:
        global_step = start
    global_step_ = global_step
    threshold = N_iters + 1
    for i in trange(start, threshold):
    ### core optimization loop ###
        i = i+global_step_
        if i == 0:
            init_nerf(graph.nerf)
            init_nerf(graph.nerf_fine)

        img_idx = torch.randperm(images.shape[0])

        if (i % args.i_img == 0 or i % args.i_novel_view == 0) and i > 0:
            ret, ray_idx, spline_poses, all_poses = graph.forward(i, img_idx, poses_num, H, W, K, args)
        else:
            ret, ray_idx, spline_poses = graph.forward(i, img_idx, poses_num, H, W, K, args)

        shape0 = img_idx.shape[0]
        interval = ray_idx.shape[0]

        # blur loss & event loss
        target_img = images[img_idx][:,ray_idx].view(-1,3)
        ret_rgb_map = ret['rgb_map'].view(shape0, args.deblur_images, interval, 3)
        ret_rgb_blur = []
        event_loss = []
        for j in range(shape0):
            event_data = event_maps[img_idx[j], :, ray_idx]
            event_loss.append(event_loss_call_blender(ret_rgb_map[j], event_data, rgb2grey, args.bin_num) * 0.005)
            for k in range(args.deblur_images):
                ret_rgb_blur.append(ret_rgb_map[j][k])
        ret_rgb_blur = torch.stack(ret_rgb_blur).view(shape0, args.deblur_images, interval, 3)
        ret_rgb_blur = torch.mean(ret_rgb_blur,dim=1)
        rgb_blur = ret_rgb_blur.reshape(-1, 3)

        event_loss = torch.mean(torch.stack(event_loss))
        img_loss_blur = img2mse(rgb_blur, target_img)
        psnr = mse2psnr(img_loss_blur)
        loss = img_loss_blur + event_loss

        if 'rgb0' in ret:
            ret_rgb_map_extra = ret['rgb0'].view(shape0, args.deblur_images, interval, 3)
            rgb_extra = ret_rgb_map_extra[:,args.deblur_images // 2].reshape(-1, 3)
            img_loss_extra = img2mse(rgb_extra, target_img)
            psnr0 = mse2psnr(img_loss_extra)
            loss = loss + img_loss_extra

        # backward
        optimizer_se3.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_se3.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        decay_rate_pose = 0.01
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in optimizer_se3.param_groups:
            param_group['lr'] = new_lrate_pose



        # evaluating
        if i%args.i_print==0:
            writer.add_scalar("loss", loss, i)
            writer.add_scalar("event_loss", event_loss, i)
            writer.add_scalar("img_loss", img_loss_blur, i)
            writer.add_scalar("img_loss_extra", img_loss_extra, i)

            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  fine_loss: {img_loss_blur.item()}, PSNR: {psnr.item()} coarse_loss: {img_loss_extra.item()} PSNR0: {psnr0.item()}")
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  event_loss: {event_loss.item()}")
            with open(print_file, 'a') as outfile:
                outfile.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss_extra.item()}, PSNR: {psnr.item()}\n")

        if i < 10:
            print('coarse_loss:', img_loss_extra.item())
            with open(print_file, 'a') as outfile:
                outfile.write(f"coarse loss: {img_loss_extra.item()}\n")

        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'graph': graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_se3': optimizer_se3.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_img == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                if args.deblur_images % 2 == 0:
                    i_render = torch.arange(i_train.shape[0]) * (args.deblur_images + 1) + args.deblur_images // 2
                else:
                    i_render = torch.arange(i_train.shape[0]) * args.deblur_images + args.deblur_images // 2
                imgs_render = render_image_test(i, graph, all_poses[i_render], H, W, K, args)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_video_test(i, graph, render_poses, H, W, K, args)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            print('Done, saving', rgbs.shape, disps.shape)

        if i % args.i_novel_view == 0 and i > 0:
            # Turn on novel view testing mode
            i_ = torch.arange(0, images.shape[0], args.llffhold-1)
            poses_test_se3_ = graph.se3.weight[i_,:6]
            model_test = novel_view_test.Model(poses_novel_se3, graph)
            graph_test = model_test.build_network(args)
            optimizer_test = model_test.setup_optimizer(args)
            for j in range(args.N_novel_view):
                ret_sharp, ray_idx_sharp, poses_sharp = graph_test.forward(i, img_idx, poses_num, H, W, K, args, novel_view=True)
                target_s_novel = imgs_gt_novel.reshape(-1, H * W, 3)[:, ray_idx_sharp]
                target_s_novel = target_s_novel.reshape(-1, 3)
                loss_sharp = img2mse(ret_sharp['rgb_map'], target_s_novel)
                psnr_sharp = mse2psnr(loss_sharp)
                if 'rgb0' in ret_sharp:
                    img_loss0 = img2mse(ret_sharp['rgb0'], target_s_novel)
                    loss_sharp = loss_sharp + img_loss0
                if j % 100==0:
                    print(j, psnr_sharp.item(), loss_sharp.item())
                optimizer_test.zero_grad()
                loss_sharp.backward()
                optimizer_test.step()
                decay_rate_sharp = 0.01
                decay_steps_sharp = args.lrate_decay * 100
                new_lrate_novel = args.pose_lrate * (decay_rate_sharp ** (j / decay_steps_sharp))
                for param_group in optimizer_test.param_groups:
                    if (j / decay_steps_sharp) <= 1.:
                        param_group['lr'] = new_lrate_novel * args.factor_pose_novel
            with torch.no_grad():
                imgs_render_novel = render_image_test(i, graph, poses_sharp, H, W, K, args, novel_view=True)

        if i % args.N_iters == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                path_pose = os.path.join(basedir, expname)
                i_render_pose = torch.arange(i_train.shape[0]) * args.deblur_images + args.deblur_images // 2
                render_poses_final = all_poses[i_render_pose]
                save_render_pose(render_poses_final, path_pose)

        global_step += 1



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
