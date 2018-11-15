package domain.models;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.Subsampling3DLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.Cnn3DToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;

import utils.FaultUtils;

/**
 * This class is used to retrieve all the different models that are available.
 */
public class ModelFactory {
	public static MultiLayerNetwork simpleOriginalCNN(int numLabels) {
		//
		// create the network configuration
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
				// user xavier initialization
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam()).list()
				// the first layer is a convolution layer with a kernel 2px high
				// and 3px wide
				.layer(0, new ConvolutionLayer.Builder(3, 2)
						// use one input channel
						.nIn(1).stride(1, 1).nOut(20).activation(new ActivationReLU()).build())
				// next use a pooling (subsampling) layer utilizing MAX-pooling
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 1).stride(2, 1)
								.build())
				// hidden layer in the densely connected network
				.layer(2, new DenseLayer.Builder().activation(new ActivationReLU())
						// number of hidden neurons
						.nOut(100).build())
				// output layer of the network using negativeloglikelihood as
				// loss function
				.layer(3, new OutputLayer.Builder(new LossNegativeLogLikelihood())
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

	public static MultiLayerNetwork deeperCNN(int numLabels) {
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).list()
				.layer(0,
						new ConvolutionLayer.Builder(3, 2).nIn(1).stride(1, 1).nOut(40).activation(new ActivationReLU())
								.build())
				.layer(1,
						new ConvolutionLayer.Builder(2, 2).nIn(40).stride(1, 1).nOut(30)
								.activation(new ActivationReLU()).build())
				.layer(2,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(3,
						new ConvolutionLayer.Builder(2, 2).nIn(30).stride(1, 1).nOut(20)
								.activation(new ActivationReLU()).build())
				.layer(4, new DenseLayer.Builder().activation(new ActivationReLU()).nOut(100).build())
				.layer(5,
						new OutputLayer.Builder(new LossNegativeLogLikelihood()).nOut(numLabels)
								.activation(new ActivationSoftmax()).build())
				.setInputType(InputType.convolutionalFlat(112, 6, 1)).backprop(true).pretrain(false).build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);

		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}

	public static MultiLayerNetwork deeperCNN(int height, int width, int nchannels, int numLabels) {
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).list()
				.layer(0,
						new ConvolutionLayer.Builder(2, 3).nIn(nchannels).stride(1, 1).nOut(40 * nchannels)
								.activation(new ActivationReLU()).build())
				.layer(1,
						new ConvolutionLayer.Builder(2, 2).nIn(40 * nchannels).stride(1, 1).nOut(30 * nchannels)
								.activation(new ActivationReLU()).build())
				.layer(2,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(3,
						new ConvolutionLayer.Builder(2, 2).nIn(30 * nchannels).stride(1, 1).nOut(20 * nchannels)
								.activation(new ActivationReLU()).build())
				.layer(4, new DenseLayer.Builder().activation(new ActivationReLU()).nOut(100 * nchannels).build())
				.layer(5,
						new OutputLayer.Builder(new LossNegativeLogLikelihood()).nOut(numLabels)
								.activation(new ActivationSoftmax()).build())
				// .setInputType(InputType.convolutional(height, width,
				// nchannels)).backprop(true).pretrain(false).build();
				.setInputType(InputType.convolutional(height, width, nchannels)).backprop(true).pretrain(false).build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, nchannels)));

		return neuralNetwork;
	}

	public static MultiLayerNetwork MKTestCNN(int height, int width, int nchannels, int numLabels) {
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.l2(0.005).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam())
				.list()
				.layer(0,
						new ConvolutionLayer.Builder(3, 3).nIn(nchannels).stride(1, 1).nOut(32).dropOut(0.5)
								.convolutionMode(ConvolutionMode.Same).activation(new ActivationReLU()).build())
				.layer(1,
						new ConvolutionLayer.Builder(2, 2).nIn(32).stride(1, 1).nOut(64).dropOut(0.5)
								.convolutionMode(ConvolutionMode.Same).activation(new ActivationReLU()).build())
				.layer(2,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(3,
						new ConvolutionLayer.Builder(2, 2).nIn(64).stride(1, 1).nOut(128).dropOut(0.5)
								.convolutionMode(ConvolutionMode.Same).activation(new ActivationReLU()).build())
				.layer(4, new DenseLayer.Builder().activation(new ActivationReLU()).nOut(1000).dropOut(0.5).build())
				.layer(5,
						new OutputLayer.Builder(new LossNegativeLogLikelihood()).nOut(numLabels)
								.activation(new ActivationSoftmax()).build())
				// .setInputType(InputType.convolutional(height, width,
				// nchannels)).backprop(true).pretrain(false).build();
				.setInputType(InputType.convolutional(height, width, nchannels)).backprop(true).pretrain(false).build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, nchannels)));

		return neuralNetwork;
	}

	public static MultiLayerNetwork deeperCNN3D(int height, int width, int nchannels, int numLabels) {
		int outDepth = 3;
		int outHeight = height / 2;
		int outWidth = width / 2;

		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).list()
				.layer(0,
						new Convolution3D.Builder(3, 2, 3).nIn(1).stride(1, 1, 1).nOut(40).padding(0, 0, 0)
								.convolutionMode(ConvolutionMode.Same).dilation(1, 1, 1)
								.activation(new ActivationReLU()).build())
				.layer(1,
						new Convolution3D.Builder(3, 2, 2).nIn(40).stride(1, 1, 1).nOut(30).padding(0, 0, 0)
								.convolutionMode(ConvolutionMode.Same).dilation(1, 1, 1)
								.activation(new ActivationReLU()).build())
				.layer(2,
						new Subsampling3DLayer.Builder().poolingType(Subsampling3DLayer.PoolingType.MAX)
								.convolutionMode(ConvolutionMode.Same).kernelSize(3, 2, 2).stride(1, 2, 2).build())
				.layer(3,
						new Convolution3D.Builder(3, 2, 2).nIn(30).stride(1, 1, 1).nOut(20).padding(0, 0, 0)
								.convolutionMode(ConvolutionMode.Same).dilation(1, 1, 1)
								.activation(new ActivationReLU()).build())
				.layer(4, new DenseLayer.Builder().activation(new ActivationReLU()).nOut(100).build())
				.layer(5,
						new OutputLayer.Builder(new LossNegativeLogLikelihood()).nOut(numLabels)
								.activation(new ActivationSoftmax()).build())
				// .setInputType(InputType.convolutional(height, width,
				// nchannels)).backprop(true).pretrain(false).build();
				// .setInputType(InputType.convolutional(height, width,
				// nchannels)).backprop(true).pretrain(false).build();
				.inputPreProcessor(4, new Cnn3DToFeedForwardPreProcessor(outDepth, outHeight, outWidth, 20, true))
				.setInputType(InputType.convolutional3D(3, height, width, 1)).backprop(true).pretrain(false).build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional3D(3, height, width, 1)));

		return neuralNetwork;
	}

	public static ComputationGraph computationGraphModel(int numLabels) {

		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutionalFlat(112, 6, 1))
				.addLayer("cnn1",
						new ConvolutionLayer.Builder(2, 3).nIn(1).stride(1, 1).nOut(40).activation(new ActivationReLU())
								.build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(2, 2).nIn(40).stride(1, 1).nOut(30)
								.activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(30).stride(1, 1).nOut(20)
								.activation(new ActivationReLU()).build(),
						"pool1")
				.addLayer("dense1", new DenseLayer.Builder().activation(new ActivationReLU()).nOut(100).build(), "cnn3")
				.addLayer("out", new OutputLayer.Builder(new LossNegativeLogLikelihood()).nIn(100).nOut(numLabels)
						.activation(new ActivationSoftmax()).build(), "dense1")
				.setOutputs("out");

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.backprop(true).pretrain(false).build());

		// initialize the network
		neuralNetwork.init();
		return neuralNetwork;
	}

	public static ComputationGraph computationGraphModel(int height, int width, int numChannels, int numLabels) {

		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1",
						new ConvolutionLayer.Builder(2, 3).nIn(numChannels).stride(1, 1).nOut(40)
								.activation(new ActivationReLU()).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(2, 2).nIn(40).stride(1, 1).nOut(30)
								.activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(30).stride(1, 1).nOut(20)
								.activation(new ActivationReLU()).build(),
						"pool1")
				.addLayer("dense1", new DenseLayer.Builder().activation(new ActivationReLU()).nOut(100).build(), "cnn3")
				.addLayer("out", new OutputLayer.Builder(new LossNegativeLogLikelihood()).nIn(100).nOut(numLabels)
						.activation(new ActivationSoftmax()).build(), "dense1")
				.setOutputs("out");

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.backprop(true).pretrain(false).build());

		// initialize the network
		neuralNetwork.init();
		return neuralNetwork;
	}

	public static ComputationGraph computationGraphModelYolo() {
		int nBoxes = 5;
		int numClasses = 14;
		double[][] priorBoxes = { { 1.08, 1.19 }, { 3.42, 4.41 }, { 6.63, 11.38 }, { 9.42, 5.11 }, { 16.62, 10.52 } };
		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;
		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(112, 6, 3))
				.addLayer("cnn1",
						new ConvolutionLayer.Builder(3, 2).nIn(3).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(40).activation(new ActivationReLU()).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(2, 2).nIn(40).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(30).activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(30).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(20).activation(new ActivationReLU()).build(),
						"pool1")
				.addLayer("insert1",
						new ConvolutionLayer.Builder(2, 1).nIn(20).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(15).activation(new ActivationReLU()).build(),
						"cnn3")
				.addLayer("pool2",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 1).stride(2, 1)
								.build(),
						"insert1")
				.addLayer("insert2",
						new ConvolutionLayer.Builder(2, 2).nIn(15).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(10).activation(new ActivationReLU()).build(),
						"pool2")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(1, 1).nIn(10).nOut(nBoxes * (5 + numClasses))
								.weightInit(WeightInit.XAVIER).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
						"insert2")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn4")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(112, 6, 3)));

		return neuralNetwork;
	}

	public static ComputationGraph KunkelPetersYolo(int height, int width, int numChannels) {

		int numClasses = 14;
		double[][] priorBoxes = FaultUtils.allPriors;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;

		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).l2(0.005)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, numChannels))
				.addLayer("cnn1",
						new ConvolutionLayer.Builder(2, 3).nIn(numChannels).stride(1, 1).nOut(64).dropOut(0.5)
								.activation(new ActivationReLU()).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(2, 2).nIn(64).stride(1, 1).nOut(128).dropOut(0.5)
								.activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(128).stride(1, 1).nOut(256).dropOut(0.5)
								.activation(new ActivationReLU()).build(),
						"pool1")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(1, 1).nIn(256).nOut(nBoxes * (5 + numClasses))
								.weightInit(WeightInit.XAVIER).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
						"cnn3")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn4")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, numChannels)));

		return neuralNetwork;
	}

	public static ComputationGraph computationGraphModelYolo(int height, int width, int channels) {
		int numClasses = 14;
		double[][] priorBoxes = FaultUtils.allPriors;
		int nBoxes = priorBoxes.length;

		INDArray priors = Nd4j.create(priorBoxes);
		double lambdaNoObj = 0.5;
		double lambdaCoord = 1.0;
		GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).graphBuilder()
				.addInputs("input").setInputTypes(InputType.convolutional(height, width, channels))
				.addLayer("cnn1", new ConvolutionLayer.Builder(2, 3).nIn(channels).stride(1, 1)
						.convolutionMode(ConvolutionMode.Same).nOut(32).activation(new ActivationReLU()).build(),
						"input")
				.addLayer("cnn2",
						new ConvolutionLayer.Builder(2, 2).nIn(32).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(64).activation(new ActivationReLU()).build(),
						"cnn1")
				.addLayer("pool1",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						"cnn2")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder(2, 2).nIn(64).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(128).activation(new ActivationReLU()).build(),
						"pool1")
				.addLayer("insert1",
						new ConvolutionLayer.Builder(1, 2).nIn(128).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(256).activation(new ActivationReLU()).build(),
						"cnn3")
				.addLayer("pool2",
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(1, 2).stride(1, 2)
								.build(),
						"insert1")
				.addLayer("insert2",
						new ConvolutionLayer.Builder(2, 2).nIn(256).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.nOut(512).activation(new ActivationReLU()).build(),
						"pool2")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder(1, 1).nIn(512).nOut(nBoxes * (5 + numClasses))
								.weightInit(WeightInit.XAVIER).stride(1, 1).convolutionMode(ConvolutionMode.Same)
								.weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(),
						"insert2")
				.addLayer("outputs",
						new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
								.boundingBoxPriors(priors).build(),
						"cnn4")
				.setOutputs("outputs").backprop(true).pretrain(false);

		ComputationGraph neuralNetwork = new ComputationGraph(graphBuilder.build());

		// initialize the network
		neuralNetwork.init();
		System.out.println(neuralNetwork.summary(InputType.convolutional(height, width, channels)));

		return neuralNetwork;
	}

	public static MultiLayerNetwork deeperPaddedCNN(int numLabels) {
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).list()
				.layer(0, new ZeroPaddingLayer(2, 2, 2, 2))// 114x8 input now
				.layer(1,
						new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(40).activation(new ActivationReLU())
								.build())
				.layer(2,
						new ConvolutionLayer.Builder(2, 2).nIn(40).stride(1, 1).nOut(30)
								.activation(new ActivationReLU()).build())
				.layer(3,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(4,
						new ConvolutionLayer.Builder(2, 2).nIn(30).stride(1, 1).nOut(20)
								.activation(new ActivationReLU()).build())
				.layer(5, new DenseLayer.Builder().activation(new ActivationReLU()).nOut(100).build())
				.layer(6,
						new OutputLayer.Builder(new LossNegativeLogLikelihood()).nOut(numLabels)
								.activation(new ActivationSoftmax()).build())
				.setInputType(InputType.convolutionalFlat(112, 6, 1)).backprop(true).pretrain(false).build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}

	public static MultiLayerNetwork simpleWeightedCNN(int numLabels, INDArray weights) {

		// create the network configuration
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
				// user xavier initialization
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new AdaDelta()).list()
				// the first layer is a convolution layer with a kernel 2px high
				// and 3px wide
				.layer(0, new ConvolutionLayer.Builder(3, 2)
						// use one input channel
						.nIn(1).stride(1, 1).nOut(20).activation(new ActivationReLU()).build())
				// next use a pooling (subsampling) layer utilizing MAX-pooling
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 1).stride(2, 1)
								.build())
				// hidden layer in the densely connected network
				.layer(2, new DenseLayer.Builder().activation(new ActivationReLU())
						// number of hidden neurons
						.nOut(100).build())
				// output layer of the network using negativeloglikelihood as
				// loss function
				.layer(3, new OutputLayer.Builder(new LossNegativeLogLikelihood(weights))
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

	public static MultiLayerNetwork simpleCNN(int numLabels) {

		// create the network configuration
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
				// user xavier initialization
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new AdaDelta()).list()
				// the first layer is a convolution layer with a kernel 2px high
				// and 3px wide
				.layer(0, new ConvolutionLayer.Builder(3, 2)
						// use one input channel
						.nIn(1).stride(1, 1).nOut(20).activation(new ActivationReLU()).build())
				// next use a pooling (subsampling) layer utilizing MAX-pooling
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 1).stride(2, 1)
								.build())
				// hidden layer in the densely connected network
				.layer(2, new DenseLayer.Builder().activation(new ActivationReLU())
						// number of hidden neurons
						.nOut(100).build())
				// output layer of the network using negativeloglikelihood as
				// loss function
				.layer(3, new OutputLayer.Builder(new LossL2())
						// use as many output neurons as there are
						// labels
						.nOut(numLabels).activation(new ActivationSigmoid()).build())
				// .layer(3,
				// new OutputLayer.Builder(new LossNegativeLogLikelihood())
				// // use as many output neurons as there are
				// // labels
				// .nOut(numLabels).activation(new ActivationSoftmax()).build())
				// the images are represented as vectors, thus the input type is
				// convolutionalFlat
				.setInputType(InputType.convolutionalFlat(112, 6, 1)).backprop(true).pretrain(false).build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}

	public static MultiLayerNetwork simpleNN(int numInputs, int numLabels) {
		int numHiddenNodes = numInputs / 2;
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new AdaDelta()).list()
				.layer(0, new OutputLayer.Builder(new LossNegativeLogLikelihood())
						// use as many output neurons as there are
						// labels
						.nIn(numInputs).nOut(numLabels).weightInit(WeightInit.XAVIER)
						.activation(new ActivationSoftmax()).build())
				.setInputType(InputType.convolutionalFlat(112, 6, 1)).pretrain(false).backprop(true).build();
		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}

	public static MultiLayerNetwork simpleDNNOne(int numInputs, int numLabels) {
		int numHiddenNodes = numInputs / 2;
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new AdaDelta()).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(numLabels).weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU).build())
				.layer(1, new OutputLayer.Builder(new LossNegativeLogLikelihood())
						// use as many output neurons as there are
						// labels
						.nIn(numLabels).nOut(numLabels).activation(new ActivationSoftmax()).build())
				.setInputType(InputType.convolutionalFlat(112, 6, 1)).pretrain(false).backprop(true).build();
		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}

	public static MultiLayerNetwork simpleDNN(int numInputs, int numLabels) {
		int numHiddenNodes = numInputs / 2;
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new AdaDelta()).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU).build())
				.layer(1,
						new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes / 2)
								.weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
				.layer(2, new OutputLayer.Builder(new LossNegativeLogLikelihood())
						// use as many output neurons as there are
						// labels
						.nOut(numLabels).activation(new ActivationSoftmax()).build())
				.setInputType(InputType.feedForward(numInputs)).pretrain(false).backprop(true).build();
		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}

	public static MultiLayerNetwork simpleDNNII(int numInputs, int numLabels) {
		int numHiddenNodes = numInputs / 2;
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new AdaDelta()).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU).build())
				.layer(1,
						new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes / 2)
								.weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
				.layer(2, new OutputLayer.Builder(new LossNegativeLogLikelihood())
						// use as many output neurons as there are
						// labels
						.nOut(numLabels).activation(new ActivationSoftmax()).build())
				.setInputType(InputType.convolutionalFlat(112, 6, 1)).pretrain(false).backprop(true).build();
		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}

	public static MultiLayerNetwork anomolyDetection(int numInputs, int numLabels) {

		// Set up network. 672 in/out (as DC data is arranged as 112x6).
		// 672 -> 225 -> 10 -> 225 -> 672

		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new AdaDelta()).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(225).weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU).build())
				.layer(1,
						new DenseLayer.Builder().nIn(225).nOut(10).weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU).build())
				.layer(2,
						new DenseLayer.Builder().nIn(10).nOut(225).weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU).build())
				.layer(3, new OutputLayer.Builder(new LossNegativeLogLikelihood())
						// use as many output neurons as there are
						// labels
						.nIn(225).nOut(numLabels).activation(new ActivationSoftmax()).build())
				.setInputType(InputType.convolutionalFlat(112, 6, 1)).pretrain(false).backprop(true).build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}

	public static MultiLayerNetwork mnistSetUp(int numInputs, int numLabels) {
		int rngSeed = 123; // random number seed for reproducibility
		double rate = 0.0015; // learning rate

		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().seed(rngSeed) // include
				// a
				// random
				// seed
				// for
				// reproducibility
				.activation(Activation.RELU).weightInit(WeightInit.XAVIER).updater(new Nesterovs(rate, 0.98))
				.l2(rate * 0.005) // regularize learning model
				.list().layer(0, new DenseLayer.Builder() // create the first input
															// layer.
						.nIn(numInputs).nOut(500).build())
				.layer(1, new DenseLayer.Builder() // create the second input
													// layer
						.nIn(500).nOut(100).build())
				.layer(2,
						new OutputLayer.Builder(new LossNegativeLogLikelihood()).activation(new ActivationSoftmax())
								.nIn(100).nOut(numLabels).build())
				.pretrain(false).backprop(true) // use backpropagation to adjust
												// weights
				.build();

		// now create the neural network from the configuration
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
		// initialize the network
		neuralNetwork.init();

		return neuralNetwork;
	}

}
