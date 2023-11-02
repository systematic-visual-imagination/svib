import os
import json
import glob

import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobSysVImDataset_GT_CLEVRTex(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size

        self.total_dirs = sorted(glob.glob(os.path.join(self.root, "*")))

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test_id':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        elif phase == 'full':
            self.total_dirs = self.total_dirs

        self.transform = transforms.ToTensor()

        self.SHAPES = ['Cone', 'Cube', 'Cylinder', 'Suzanne', 'Icosahedron', 'NewellTeapot', 'Sphere', 'Torus']

        self.SIZES = [
            1.,
            1.5,
            2.,
        ]

        self.MATERIALS = [
            'data/materials/PoliigonBricksFlemishRed001.blend',
            'data/materials/PoliigonBricksPaintedWhite001.blend',
            'data/materials/PoliigonCarpetTwistNatural001.blend',
            'data/materials/PoliigonChainmailCopperRoundedThin001.blend',
            'data/materials/PoliigonCityStreetAsphaltGenericCracked002.blend',
            'data/materials/PoliigonCityStreetRoadAsphaltTwoLaneWorn001.blend',
            'data/materials/PoliigonCliffJagged004.blend',
            'data/materials/PoliigonCobblestoneArches002.blend',
            'data/materials/PoliigonConcreteWall001.blend',
            'data/materials/PoliigonFabricDenim003.blend',
            'data/materials/PoliigonFabricFleece001.blend',
            'data/materials/PoliigonFabricLeatherBuffaloRustic001.blend',
            'data/materials/PoliigonFabricRope001.blend',
            'data/materials/PoliigonFabricUpholsteryBrightAnglePattern001.blend',
            'data/materials/PoliigonGroundClay002.blend',
            'data/materials/PoliigonGroundDirtForest014.blend',
            'data/materials/PoliigonGroundDirtRocky002.blend',
            'data/materials/PoliigonGroundForest003.blend',
            'data/materials/PoliigonGroundForest008.blend',
            'data/materials/PoliigonGroundForestMulch001.blend',
            'data/materials/PoliigonGroundForestRoots001.blend',
            'data/materials/PoliigonGroundMoss001.blend',
            'data/materials/PoliigonGroundSnowPitted003.blend',
            'data/materials/PoliigonGroundTireTracks001.blend',
            'data/materials/PoliigonInteriorDesignRugStarryNight001.blend',
            'data/materials/PoliigonMarble13.blend',
            'data/materials/PoliigonMarble062.blend',
            'data/materials/PoliigonMetalCorrodedHeavy001.blend',
            'data/materials/PoliigonMetalCorrugatedIronSheet002.blend',
            'data/materials/PoliigonMetalDesignerWeaveSteel002.blend',
            'data/materials/PoliigonMetalPanelRectangular001.blend',
            'data/materials/PoliigonMetalSpottyDiscoloration001.blend',
            'data/materials/PoliigonPlaster07.blend',
            'data/materials/PoliigonPlaster17.blend',
            'data/materials/PoliigonRoadCityWorn001.blend',
            'data/materials/PoliigonRoofTilesTerracotta004.blend',
            'data/materials/PoliigonRustMixedOnPaint012.blend',
            'data/materials/PoliigonRustPlain007.blend',
            'data/materials/PoliigonSolarPanelsPolycrystallineTypeBFramedClean001.blend',
            'data/materials/PoliigonStoneBricksBeige015.blend',
            'data/materials/PoliigonStoneMarbleCalacatta004.blend',
            'data/materials/PoliigonTerrazzoVenetianMatteWhite001.blend',
            'data/materials/PoliigonTiles05.blend',
            'data/materials/PoliigonTilesMarbleChevronCreamGrey001.blend',
            'data/materials/PoliigonTilesMarbleSageGreenBrickBondHoned001.blend',
            'data/materials/PoliigonTilesOnyxOpaloBlack001.blend',
            'data/materials/PoliigonTilesRectangularMirrorGray001.blend',
            'data/materials/PoliigonWallMedieval003.blend',
            'data/materials/PoliigonWaterDropletsMixedBubbled001.blend',
            'data/materials/PoliigonWoodFineDark004.blend',
            'data/materials/PoliigonWoodFlooring044.blend',
            'data/materials/PoliigonWoodFlooring061.blend',
            'data/materials/PoliigonWoodFlooringMahoganyAfricanSanded001.blend',
            'data/materials/PoliigonWoodFlooringMerbauBrickBondNatural001.blend',
            'data/materials/PoliigonWoodPlanks028.blend',
            'data/materials/PoliigonWoodPlanksWorn33.blend',
            'data/materials/PoliigonWoodQuarteredChiffon001.blend',
            'data/materials/WhiteMarble.blend'
        ]

        self.X_MIN = -4
        self.X_MAX = 4
        self.Y_MIN = -4
        self.Y_MAX = 4
        self.ROT_MIN = 0
        self.ROT_MAX = 360

        self.LAMP_CAM_MIN = -10.
        self.LAMP_CAM_MAX = 10.

    def __len__(self):
        return len(self.total_dirs)

    def __getitem__(self, idx):
        source_path = os.path.join(self.total_dirs[idx], 'source.json')
        with open(source_path) as source_file:
            source_info = json.load(source_file)

        discrete_factors = []
        float_factors = []
        for obj_info in source_info['objects']:

            obj_float_factors = [
                (obj_info['3d_coords'][0] - self.X_MIN) / (self.X_MAX - self.X_MIN),
                (obj_info['3d_coords'][1] - self.Y_MIN) / (self.Y_MAX - self.Y_MIN),
                (obj_info['rotation'] - self.ROT_MIN) / (self.ROT_MAX - self.ROT_MIN),

                (source_info['Lamp_Key'][0] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Key'][1] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Key'][2] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),

                (source_info['Lamp_Back'][0] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Back'][1] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Back'][2] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),

                (source_info['Lamp_Fill'][0] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Fill'][1] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Fill'][2] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),

                (source_info['Camera'][0] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Camera'][1] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Camera'][2] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
            ]

            obj_discrete_factors = [
                self.SHAPES.index(obj_info['shape']),
                self.SIZES.index(obj_info['size']),
                self.MATERIALS.index(obj_info['material']),
                self.MATERIALS.index(source_info['ground_material']),
            ]

            discrete_factors.append(obj_discrete_factors)
            float_factors.append(obj_float_factors)

        discrete_factors = torch.Tensor(discrete_factors).long()  # N, M_discrete
        float_factors = torch.Tensor(float_factors).float()  # N, M_float

        img_loc_target = os.path.join(self.total_dirs[idx], 'target.png')
        image_target = Image.open(img_loc_target).convert("RGB")
        image_target = image_target.resize((self.img_size, self.img_size))
        tensor_image_target = self.transform(image_target)

        return discrete_factors, float_factors, tensor_image_target


class GlobSysVImDataset_GT_CLEVR(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size

        self.total_dirs = sorted(glob.glob(os.path.join(self.root, "*")))

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test_id':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        elif phase == 'full':
            self.total_dirs = self.total_dirs

        self.transform = transforms.ToTensor()

        self.SHAPES = ['SmoothCube_v2', 'Sphere', 'SmoothCylinder', 'Suzanne']

        self.SIZES = [
            1.,
            1.5,
            2.,
        ]

        self.MATERIALS = [
            "Rubber",
            "MyMetal"
        ]

        self.COLORS = [
            (1., 0., 0., 1.),
            (0., 1., 0., 1.),
            (0., 0., 1., 1.),
            (0., 1., 1., 1.),
            (1., 0., 1., 1.),
            (1., 1., 0., 1.),
        ]

        self.X_MIN = -4
        self.X_MAX = 4
        self.Y_MIN = -4
        self.Y_MAX = 4
        self.ROT_MIN = 0
        self.ROT_MAX = 360
        self.COLOR_MIN = 0
        self.COLOR_MAX = 255

        self.LAMP_CAM_MIN = -10.
        self.LAMP_CAM_MAX = 10.

    def __len__(self):
        return len(self.total_dirs)

    def __getitem__(self, idx):
        source_path = os.path.join(self.total_dirs[idx], 'source.json')
        with open(source_path) as source_file:
            source_info = json.load(source_file)

        discrete_factors = []
        float_factors = []
        for obj_info in source_info['objects']:

            obj_float_factors = [
                (obj_info['3d_coords'][0] - self.X_MIN) / (self.X_MAX - self.X_MIN),
                (obj_info['3d_coords'][1] - self.Y_MIN) / (self.Y_MAX - self.Y_MIN),
                (obj_info['rotation'] - self.ROT_MIN) / (self.ROT_MAX - self.ROT_MIN),

                (source_info['Lamp_Key'][0] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Key'][1] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Key'][2] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),

                (source_info['Lamp_Back'][0] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Back'][1] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Back'][2] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),

                (source_info['Lamp_Fill'][0] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Fill'][1] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Lamp_Fill'][2] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),

                (source_info['Camera'][0] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Camera'][1] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
                (source_info['Camera'][2] - self.LAMP_CAM_MIN) / (self.LAMP_CAM_MAX - self.LAMP_CAM_MIN),
            ]

            obj_discrete_factors = [
                self.COLORS.index(tuple(obj_info['color'])),
                self.SHAPES.index(obj_info['shape']),
                self.SIZES.index(obj_info['size']),
                self.MATERIALS.index(obj_info['material']),
            ]

            discrete_factors.append(obj_discrete_factors)
            float_factors.append(obj_float_factors)

        discrete_factors = torch.Tensor(discrete_factors).long()  # N, M_discrete
        float_factors = torch.Tensor(float_factors).float()  # N, M_float

        img_loc_target = os.path.join(self.total_dirs[idx], 'target.png')
        image_target = Image.open(img_loc_target).convert("RGB")
        image_target = image_target.resize((self.img_size, self.img_size))
        tensor_image_target = self.transform(image_target)

        return discrete_factors, float_factors, tensor_image_target


class GlobSysVImDataset_GT_DSPRITES(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size

        self.total_dirs = sorted(glob.glob(os.path.join(self.root, "*")))

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test_id':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        elif phase == 'full':
            self.total_dirs = self.total_dirs

        self.transform = transforms.ToTensor()

        self.COLORS = [[0, 255, 0], [255, 0, 255], [0, 127, 255], [255, 127, 0]]

        self.SHAPES = ["circle", "triangle", "square", "star_4"]
        
        self.SIZES = [0.125, 0.225, 0.325, 0.425]

        self.X_MIN = 0
        self.X_MAX = 1
        self.Y_MIN = 0
        self.Y_MAX = 1
        self.ROT_MIN = 0
        self.ROT_MAX = 360

    def __len__(self):
        return len(self.total_dirs)

    def __getitem__(self, idx):
        source_path = os.path.join(self.total_dirs[idx], 'source.json')
        with open(source_path) as source_file:
            source_info = json.load(source_file)

        discrete_factors = []
        float_factors = []
        for obj_info in source_info['objects']:

            obj_float_factors = [
                (obj_info['2d_coords'][0] - self.X_MIN) / (self.X_MAX - self.X_MIN),
                (obj_info['2d_coords'][1] - self.Y_MIN) / (self.Y_MAX - self.Y_MIN),
                #(obj_info['rotation'] - self.ROT_MIN) / (self.ROT_MAX - self.ROT_MIN),
            ]

            obj_discrete_factors = [
                self.COLORS.index(obj_info['color']),
                self.SHAPES.index(obj_info['shape']),
                self.SIZES.index(obj_info['size']),
            ]

            discrete_factors.append(obj_discrete_factors)
            float_factors.append(obj_float_factors)

        discrete_factors = torch.Tensor(discrete_factors).long()  # N, M_discrete
        float_factors = torch.Tensor(float_factors).float()  # N, M_float

        img_loc_target = os.path.join(self.total_dirs[idx], 'target.png')
        image_target = Image.open(img_loc_target).convert("RGB")
        image_target = image_target.resize((self.img_size, self.img_size))
        tensor_image_target = self.transform(image_target)

        return discrete_factors, float_factors, tensor_image_target
