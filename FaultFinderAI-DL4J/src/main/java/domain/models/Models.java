package domain.models;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;

import utils.FaultUtils;

/**
 * This class is used to retrieve all the different models that are available.
 */
public class Models {
	public static MultiLayerNetwork KunkelPeters(int height, int width, int numChannels, int numClasses) {
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).list()
				.layer(0,
						new ConvolutionLayer.Builder(2, 3).nIn(numChannels).stride(1, 1).nOut(40)
								.activation(new ActivationReLU()).build())
				.layer(1,
						new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(30).activation(new ActivationReLU())
								.build())
				.layer(2,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(3,
						new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(20).activation(new ActivationReLU())
								.build())
				.layer(4,
						new DenseLayer.Builder().activation(new ActivationReLU()).nOut(100 * numChannels).dropOut(0.5)
								.build())
				.layer(5,
						new OutputLayer.Builder(new LossNegativeLogLikelihood()).nOut(numClasses)
								.activation(new ActivationSoftmax()).build())
				// .setInputType(InputType.convolutional(height, width,
				// nchannels)).backprop(true).pretrain(false).build();
				// 40, 30, 20, 100 was original
				.setInputType(InputType.convolutional(height, width, numChannels)).build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static MultiLayerNetwork deeperDNNMultiClass(int height, int width, int nchannels, int numLabels) {
		double learningRate = 1e-4;
		double l2Rate = 0.00001;
		int seed = 123;
		double lReLUVal = 0.1;
		int numHiddenNodes = height * width / 2;
		IUpdater updater = new Adam(learningRate);
		// .nIn(height * width)
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().seed(seed)
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(updater).l2(l2Rate).activation(Activation.IDENTITY).list()
				.layer(0,
						new DenseLayer.Builder().nOut(numHiddenNodes).weightInit(WeightInit.XAVIER)
								.activation(new ActivationLReLU(lReLUVal)).build())
				// .layer(1,
				// new DenseLayer.Builder().nOut(numHiddenNodes /
				// 2).weightInit(WeightInit.XAVIER)
				// .activation(new ActivationLReLU(lReLUVal)).build())
				.layer(1,
						new OutputLayer.Builder(new LossBinaryXENT()).nOut(numLabels)
								.activation(new ActivationSigmoid()).build())
				// .layer(2, new OutputLayer.Builder(new
				// LossMultiLabel()).nOut(numLabels).build())
				.setInputType(InputType.convolutional(height, width, nchannels)).build();
		// LossBinaryXENT .activation(new ActivationSigmoid()) LossMultiLabel
		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, nchannels)));

		return neuralNetwork;
	}

	public static MultiLayerNetwork wireMultiClass(int height, int width, int nchannels, int numLabels) {
		double learningRate = 1e-4;
		double l2Rate = 0.00001;
		int seed = 123;
		double lReLUVal = 0.1;
		int numHiddenNodes = height * width / 2;
		IUpdater updater = new Adam(learningRate);
		// .nIn(height * width)
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().seed(seed)
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(updater).l2(l2Rate).activation(Activation.IDENTITY).list()
				.layer(0,
						new DenseLayer.Builder().nOut(numHiddenNodes).weightInit(WeightInit.XAVIER)
								.activation(new ActivationLReLU(lReLUVal)).build())
				.layer(1,
						new DenseLayer.Builder().nOut(numHiddenNodes / 2).weightInit(WeightInit.XAVIER)
								.activation(new ActivationLReLU(lReLUVal)).build())
				.layer(2,
						new DenseLayer.Builder().nOut(numHiddenNodes).weightInit(WeightInit.XAVIER)
								.activation(new ActivationLReLU(lReLUVal)).build())
				.layer(3,
						new OutputLayer.Builder(new LossBinaryXENT()).nOut(numLabels)
								.activation(new ActivationSigmoid()).build())
				// .layer(2, new OutputLayer.Builder(new
				// LossMultiLabel()).nOut(numLabels).build())
				.setInputType(InputType.convolutional(height, width, nchannels)).build();
		// LossBinaryXENT .activation(new ActivationSigmoid()) LossMultiLabel
		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, nchannels)));

		return neuralNetwork;
	}

	public static void main(String[] args) {
		/**
		 * just to see architecture
		 */
		double[][] priorBoxes = FaultUtils.allPriors;
		// Models.CLASModel(216, 112, 3);

		// Models.test(216, 112, 3);
		// Models.KunkelPeterModel(6, 112, 3, 2);
		ComputationGraph test = Models.wireMultiClass(6, 112, 1, 112).toComputationGraph();
		System.out.println(test.summary(InputType.convolutional(6, 112, 1)));

	}

}
