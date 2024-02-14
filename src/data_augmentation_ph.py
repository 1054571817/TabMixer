from monai.transforms import (
    Compose
)

from monai.transforms import (
    EnsureChannelFirstD,
    CenterSpatialCropD,
    CropForegroundD,
    EnsureTypeD,
    LoadImageD,
    RandAdjustContrastD,
    RandGaussianNoiseD,
    RandScaleIntensityD,
    RandShiftIntensityD,
    RandAffineD,
    RandRotateD,
    ResizeD,
    ScaleIntensityRangePercentilesD,
    SpatialPadD,
    ToDeviceD,
)


class Transforms():
    def __init__(self,
                 args,
                 device: str = 'cpu',
                 ) -> None:
        keys = ["image"]

        self.val_transform = Compose(
            [
                # INITAL SETUP
                LoadImageD(keys=keys, reader='NumpyReader', image_only=False),
                EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                ToDeviceD(keys=keys, device=device),
                # GEOMETRIC - NON-RANDOM - PREPROCESING
                # NON-RANDOM - perform on GPU
                CenterSpatialCropD(keys=keys, roi_size=args.data.augmentation.spatial_crop_size),
                SpatialPadD(keys=keys, spatial_size=args.data.augmentation.padding_size),
                ResizeD(keys=keys, spatial_size=args.data.augmentation.resize_size, mode="bilinear"),
                # INTENSITY - NON-RANDOM - PREPROCESING
                ##image
                ScaleIntensityRangePercentilesD(keys="image",
                                                lower=0,
                                                upper=100,
                                                b_min=0.0,
                                                b_max=1.0,
                                                clip=True),
                ToDeviceD(keys=keys, device=device)
            ]
        )
        self.train_transform = Compose(
            [
                # INITAL SETUP
                LoadImageD(keys=keys, reader='NumpyReader', image_only=False),
                EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                ToDeviceD(keys=keys, device=device),
                EnsureTypeD(keys=keys, data_type="tensor", device=device),
                # GEOMETRIC - NON-RANDOM - PREPROCESING
                RandRotateD(keys=keys, range_x=1, range_y=0, range_z=0,
                            mode="bilinear", prob=0.7),
                RandAffineD(keys=keys, prob=0.9, translate_range=(
                    0, args.data.augmentation.spatial_crop_size[1] // 4,
                    args.data.augmentation.spatial_crop_size[2] // 4),
                            scale_range=[(0, 0), (0, 0), (0, 0)],
                            mode="bilinear", padding_mode='zeros'),
                CenterSpatialCropD(keys=keys, roi_size=args.data.augmentation.spatial_crop_size),
                SpatialPadD(keys=keys, spatial_size=args.data.augmentation.padding_size),
                ResizeD(keys=keys, spatial_size=args.data.augmentation.resize_size, mode="bilinear"),
                # INTENSITY - NON-RANDOM - PREPROCESING
                ##image
                ScaleIntensityRangePercentilesD(keys="image",
                                                lower=0,
                                                upper=100,
                                                b_min=0.0,
                                                b_max=1.0,
                                                clip=True),
                # GEOMETRIC - RANDOM - DATA AUGMENTATION
                ToDeviceD(keys=keys, device=device),
                RandAdjustContrastD(keys="image",
                                    gamma=(0.5, 2.0),
                                    prob=0.25),
                RandShiftIntensityD(keys="image", offsets=0.20, prob=0.5),
                RandScaleIntensityD(keys="image", factors=0.15, prob=0.5),
                RandGaussianNoiseD(keys="image", prob=0.5, std=0.01),
                # FINAL CHECK
                EnsureTypeD(keys=keys, data_type="tensor", device=device)
            ]
        )
