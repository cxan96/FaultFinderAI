package client;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * This class is used to retrieve all the different models that are available.
 */
public class ModelFactory {

    
    public static MultiLayerNetwork simpleCNN(int numLabels) {
	
	// create the network configuration
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
            // user xavier initialization
            .weightInit(WeightInit.XAVIER)
            // use stochastic gradient descent
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            // also use momentum (first parameter)
            .updater(new Nesterovs(0.1, 0.01))
            .list()
            // the first layer is a convolution layer with a kernel size of 2x2 pixels
            .layer(0, new ConvolutionLayer.Builder(2, 2)
                   // use one input channel
                   .nIn(1)
                   .stride(1, 1)
                   .nOut(10)
                   .activation(new ActivationReLU())
                   .build())
            // next use a pooling (subsampling) layer utilizing MAX-pooling
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                   .kernelSize(2, 2)
                   .stride(1, 1)
                   .build())
            // hidden layer in the densely connected network
            .layer(2, new DenseLayer.Builder()
                   .activation(new ActivationReLU())
		   // number of hidden neurons
                   .nOut(100)
                   .build())
            // output layer of the network using L2 as loss function
            .layer(3, new OutputLayer.Builder(new LossL2())
                   // use as many output neurons as there are labels
                   .nOut(numLabels)
                   .activation(new ActivationSoftmax())
                   .build())
            // the images are represented as vectors, thus the input type is convolutionalFlat
            .setInputType(InputType.convolutionalFlat(112,6, 1))
            .backprop(true)
            .pretrain(false)
            .build();

        // now create the neural network from the configuration
        MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
        // initialize the network
        neuralNetwork.init();

	return neuralNetwork;
    }
}
