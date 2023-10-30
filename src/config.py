import torch


class TrainGlobalConfig:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_dir_tdcsfog = "./data/train/tdcsfog/"
    train_dir_defog = "./data/train/defog/"

    test_dir = "./data/test/"

    metadata_tdcsfog = "./data/tdcsfog_metadata.csv"
    metadata_defog = "./data/defog_metadata.csv"

    window_size = 1000
    window_future = 50
    window_past = window_size - window_future
    '''
    滑动窗口是处理时序数据的常见方法 在时间序列数据中捕获局部模式，并为机器学习模型提供固定大小的输入
    950 past→50 future
    一个优点是提供了一种结构化的方法来捕获时序数据中的模式和依赖性
    选择合适的窗口大小是关键。太小的窗口可能无法捕获足够的信息，而太大的窗口可能会导致计算上的挑战，并可能包括不必要的信息。
    '''

    folds = 5
    num_workers = 4
    batch_size = 1024
    n_epochs = 1 #?
    lr = 0.001

    folder = "weights/FoG/"
    verbose = True
    verbose_step = 1

    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode="min", factor=0.7, patience=10, verbose=False, min_lr=0.000001
    )
    #基于验证损失的学习率衰减
