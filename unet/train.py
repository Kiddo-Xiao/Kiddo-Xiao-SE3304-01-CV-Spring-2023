from processor import *


if __name__ == '__main__':
    # m = Processor(data_path='./dataset')
    # m.run()
    m = Processor()
    m.extract(model_path='./model/thyroid-50.pth')
    m.predict('./dataset/images-ori/132.png', './outputs/132.png', threshold=0.5)
