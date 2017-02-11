# -*- coding: cp932 -*-
# py2-3 間のprint互換性
from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from matplotlib import pyplot as plt
import numpy as np

_resume = True
_resumeNum = 21
# network definition
class NN(chainer.Chain):
    def __init__(self, n_units, n_out, filtersize=(3,3)):
        self.train = True
        outputCh = 13;
        """
            L.Convolution2D(inputCh, outputCh, filtersize)
                inputCh: 入力チャンネル数。白黒->1,RGB->3
                outputCh: 出力チャンネル数。
                filtersize: フィルターの大きさ。3*3->3, 3*4->(3,4)
                stride=: フィルタの移動幅。小さいほうが良い。
                pad=: 画像周辺におくパディング幅。通例floor(H/2)
        """
        super(NN, self).__init__(
            conv1 = F.Convolution2D(1, outputCh, filtersize),
            conv2 = F.Convolution2D(outputCh, outputCh, filtersize),
            l3 = L.Linear(None, n_units),
            l4 = L.Linear(None, n_out),
        )
        # x->conv->relu->conv->relu->all(dropout)->relu->all->y
    def __call__(self, x, t):
        """ 
        F.max_pooling_2d(x, size, stride)
            x: input
            size: プーリング領域のサイズ
            stride: フィルタの移動幅。通常2以上
        """
        x = F.reshape(x,(100,1,28,28))
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=3)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=3)
        h = F.dropout(F.relu(self.l3(h)), train=self.train)
        y = self.l4(h)
        loss = F.softmax_cross_entropy(y,t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(y,t)}, self)
        return loss

    def calc(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=3)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=3)
        h = F.dropout(F.relu(self.l3(h)), train=self.train)
        return self.l4(h)

class TestEvaluator(extensions.Evaluator):
    def __init__(self, test_iter, model, trainer):
        super(TestEvaluator, self).__init__(test_iter, model)
        self.trainer = trainer
    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestEvaluator, self).evaluate()
        model.train = True
        acc = ret['validation/main/accuracy']
        if acc > 0.99:
            self.trainer.updater.get_optimizer('main').rho = 0.1
        elif acc > 0.985:
            self.trainer.updater.get_optimizer('main').rho = 0.3
        elif acc > 0.95:
            self.trainer.updater.get_optimizer('main').rho = 0.5
        return ret

def main():
    model = NN(1000,10)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')

    # 1epoch 毎に評価
    trainer.extend(TestEvaluator(test_iter, model, trainer))

    trainer.extend(extensions.dump_graph('main/loss'))

    #snapshot
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    #log
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    
    #progress bar
    trainer.extend(extensions.ProgressBar())

    # resume
    if _resume:
        chainer.serializers.load_npz('./result_Adam_0203/snapshot_iter_'+str(_resumeNum*600-600), trainer)

    #trainer.run()
    if _resumeNum == 21: 
        chainer.serializers.load_npz('./result_Adam_0203/snapshot_iter_12000', trainer)
        p = trainer.updater.get_optimizer('main').target
        from sklearn.externals import joblib
        joblib.dump(p, "mnistClassifier")
    
if __name__ == '__main__':
    main()






