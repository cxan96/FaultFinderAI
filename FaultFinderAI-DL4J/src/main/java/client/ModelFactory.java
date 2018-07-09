package client;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;

/**
 * This class is used to retrieve all the different models that are available.
 */
public class ModelFactory {

	public static MultiLayerNetwork simpleCNN(int numLabels) {

		// create the network configuration
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
				// user xavier initialization
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new AdaDelta()).list()
				// the first layer is a convolution layer with a kernel 2px high
				// and 3px wide
				.layer(0,
						new ConvolutionLayer.Builder(3, 2)
								// use one input channel
								.nIn(1).stride(1, 1).nOut(20).activation(new ActivationReLU()).build())
				// next use a pooling (subsampling) layer utilizing MAX-pooling
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(1, 2).stride(1, 1)
								.build())
				// hidden layer in the densely connected network
				.layer(2,
						new DenseLayer.Builder().activation(new ActivationReLU())
								// number of hidden neurons
								.nOut(100).build())
				// output layer of the network using negativeloglikelihood as
				// loss function
				.layer(3,
						new OutputLayer.Builder(new LossNegativeLogLikelihood())
								// use as many output neurons as there are
								// labels
								.nOut(numLabels).activation(new ActivationSoftmax()).build())
				// the images are represented as vectors, thus the input type is
				// convolutionalFlat
				.setInputType(InputType.convolutionalFlat(112, 6, 1)).backprop(true).pretrain(false).build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}
}
