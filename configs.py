params = {
            'dataset'    : 'dSprites',
            'batch_size' : 144,
            'data_path'  : './dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
            'img_size'   : (64, 64),

            'model_name'  : 'bcvae2D',
            'sch_gamma'   : 0.95,
            'lr'    : 5e-3,
            'EPOCH' : 60,
            'gamma' : 100,
            'beta'  : 7,
            'max_capacity'     : 25,
            'Capacity_max_iter': 1e5,
            'optim_num': 2,

            'results_path': "./results/",
            'models_path' : "models/",

}