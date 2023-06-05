from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import numpy as np







if __name__ == '__main__':    
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset120_ThyroidSegmentation\\nnUNetTrainer__nnUNetPlans__2d'),
        use_folds=("all",),
        checkpoint_name='checkpoint_final.pth',
    )
    # variant 1: give input and output folders
    # predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
    #                              join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
    #                              save_probabilities=False, overwrite=False,
    #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # indir = join(nnUNet_raw, 'Dataset120_ThyroidSegmentation\\imagesTs')
    # outdir = join(nnUNet_raw, 'Dataset120_ThyroidSegmentation\\output')
    # predictor.predict_from_files([[join(indir, 'THYROID_000_0000.png')], 
    #                               [join(indir, 'THYROID_001_0000.png')]],
    #                              [join(outdir, '0.png'),
    #                               join(outdir, '1.png')],
    #                              save_probabilities=False, overwrite=False,
    #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    # predict a single numpy array
    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset120_ThyroidSegmentation\\imagesTs\\THYROID_000_0000.png')])
    # img = img[0, 0, :, :]
    # img = np.stack([img, img, img])
    print(img.shape)
    print(props['spacing'])
    props['spacing'] = [0, 0]
    ret = predictor.predict_single_npy_array(img, props, None, None, False)