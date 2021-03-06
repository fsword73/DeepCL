cdef class Adagrad(Trainer):
    cdef cDeepCL.Adagrad *thisptr
    def __cinit__( self, DeepCL cl, learningRate, momentum=0.0 ):
        self.thisptr = new cDeepCL.Adagrad(cl.thisptr)
        self.thisptr.setLearningRate(learningRate)
        self.baseptr = self.thisptr
    def __dealloc__(self):
        del self.thisptr
    def setLearningRate(self, float learningRate):
        self.thisptr.setLearningRate(learningRate)
    def train(self, NeuralNet net, TrainingContext context,
        float[:] inputdata, float[:] expectedOutput ):
        cdef cDeepCL.BatchResult result = self.thisptr.train(
            net.thisptr, context.thisptr, &inputdata[0], &expectedOutput[0])
        return result.getLoss()
    def trainFromLabels(self, NeuralNet net, TrainingContext context,
        float[:] inputdata, int[:] labels):
        cdef cDeepCL.BatchResult result = self.thisptr.trainFromLabels(
            net.thisptr, context.thisptr, &inputdata[0], &labels[0])
        return ( result.getLoss(), result.getNumRight() )

